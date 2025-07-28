import os
import json
import asyncio
import logging


from typing import List, Generator, Dict, Any
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from concurrent.futures import ThreadPoolExecutor
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from app.config.settings import settings
from app.pipelines.base import BaseLoader, BaseEmbedder, BaseVectorStore, BaseLLM
from app.prompts.manager import PromptManager

from app.processing.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class LocalDocxLoader(BaseLoader):
    """
    Loads and processes .docx files using DocumentProcessor and creates two types
    of documents for ingestion:
    1. A single 'toc' document representing the entire file's structure.
    2. Multiple 'content_chunk' documents for each text section.
    """
    def __init__(self):
        self.processor = DocumentProcessor()
        logger.info("LocalDocxLoader initialized with custom DocumentProcessor.")

    def _create_toc_string(self, nodes: List[Dict], indent: int = 0) -> str:
        """Helper to create a clean string representation of the TOC for embedding."""
        parts = []
        for node in nodes:
            parts.append("  " * indent + node.get("topic", "Unnamed"))
            if node.get("subtopics"):
                parts.append(self._create_toc_string(node["subtopics"], indent + 1))
        return "\n".join(parts)

    def _flatten_and_create_content_documents(
        self,
        nodes: List[Dict[str, Any]],
        source_filename: str,
        parent_path: str = ""
    ) -> Generator[Document, None, None]:
        """Recursively traverses the content tree and yields 'content_chunk' Documents."""
        for node in nodes:
            current_topic = node.get("topic", "Unnamed Section")
            new_path = f"{parent_path} > {current_topic}" if parent_path else current_topic
            content = node.get("content")

            if not content:
                logger.warning(f"No content found for topic '{new_path}' in {source_filename}. Skipping content chunk.")
            else:
                yield Document(
                    page_content=content,
                    metadata={
                        "type": "content_chunk",
                        "source": source_filename,
                        "topic": current_topic,
                        "path": new_path,
                        "page_number": node.get("page_number")
                    }
                )

            # Recurse into subtopics
            subtopics = node.get("subtopics", [])
            if subtopics:
                yield from self._flatten_and_create_content_documents(subtopics, source_filename, new_path)

    def load(self, data_path: str) -> Generator[Document, None, None]:
        """Finds .docx files, processes them, and yields Documents of type 'toc' and 'content_chunk'."""
        logger.info(f"Starting to load and process documents from: {data_path}")
        if not os.path.isdir(data_path):
            logger.error(f"Data path '{data_path}' is not a valid directory.")
            return

        for filename in os.listdir(data_path):
            if filename.endswith(".docx") and not filename.startswith('~'):
                file_path = os.path.join(data_path, filename)
                logger.info(f"Processing document: {file_path}")
                try:
                    with open(file_path, 'rb') as f:
                        processed_data = self.processor.process(f)
                    
                    toc_structure = processed_data.get('toc', [])
                    content_structure = processed_data.get('content', [])

                    if not toc_structure:
                        logger.warning(f"No TOC structure found in {filename}. Skipping file.")
                        continue

                    toc_string_for_embedding = self._create_toc_string(toc_structure)
                    policy_title = content_structure[0].get("topic") if content_structure else filename
                    
                    yield Document(
                        page_content=toc_string_for_embedding,
                        metadata={
                            "type": "toc",
                            "source": filename,
                            "policy_title": policy_title,
                            "full_toc_json": json.dumps(toc_structure)
                        }
                    )

                    yield from self._flatten_and_create_content_documents(
                        nodes=content_structure,
                        source_filename=filename
                    )

                except Exception as e:
                    logger.error(f"Failed to process {filename}. Error: {e}", exc_info=True)
                    continue


class LocalEmbedder(BaseEmbedder):
    def __init__(self):
        self.client = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_HOST
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_documents(texts)

    @property
    def vector_size(self) -> int:
        return settings.VECTOR_SIZE

class LocalVectorStore(BaseVectorStore):
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)

    def setup(self, collection_name: str, vector_size: int):
        try:
            self.client.get_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' already exists.")
        except Exception:
            print(f"Collection '{collection_name}' not found. Creating...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
            print("Collection created.")

    def upsert(self, documents: List[Document], collection_name: str):
        self.client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=i,  # In a real system, use a better ID (e.g., hash of content)
                    vector=doc.metadata["embedding"],
                    payload={k: v for k, v in doc.metadata.items() if k != "embedding"}
                ) for i, doc in enumerate(documents)
            ],
            wait=True
        )

class LocalLLM(BaseLLM):
    def __init__(self, prompt_manager: PromptManager, max_workers: int = 4):
        self.client = ChatOllama(model=settings.LLM_MODEL, base_url=settings.OLLAMA_HOST)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            self.main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = None

        self.summary_chain = prompt_manager.get_prompt("generate_summary") | self.client
        self.keyword_chain = prompt_manager.get_prompt("generate_keywords") | self.client


    def _generate_for_one_text(self, text: str) -> Dict[str, Any]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)        
        try:
            summary = self.summary_chain.invoke({"text": text}).content
            keywords_str = self.keyword_chain.invoke({"text": text}).content
            keywords = [k.strip() for k in keywords_str.split(",")]
            
            return {"summary": summary, "keywords": keywords}
        finally:
            loop.close()



    async def ainvoke_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Implements the async contract by submitting sync jobs to a thread pool.
        """
        if self.main_loop is None:
            self.main_loop = asyncio.get_running_loop()

        futures = [
            self.main_loop.run_in_executor(self.executor, self._generate_for_one_text, text)
            for text in texts
        ]
        results = await asyncio.gather(*futures)
        return [res for res in results if res is not None]



