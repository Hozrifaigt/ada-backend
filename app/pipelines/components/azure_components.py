import logging
import asyncio


from typing import List, Generator, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch

from app.config.settings import settings
from app.pipelines.base import BaseLoader, BaseEmbedder, BaseVectorStore, BaseLLM

from app.prompts.manager import PromptManager 

logger = logging.getLogger(__name__)

class AzureBlobLoader(BaseLoader):
    """Loads .docx files from an Azure Blob Storage container."""
    def __init__(self):
        if not settings.AZURE_STORAGE_CONNECTION_STRING or not settings.AZURE_STORAGE_CONTAINER_NAME:
            raise ValueError("Azure Storage settings are not configured.")
        
        self.loader = AzureBlobStorageContainerLoader(
            conn_str=settings.AZURE_STORAGE_CONNECTION_STRING,
            container=settings.AZURE_STORAGE_CONTAINER_NAME,
            prefix=".docx" # Only load docx files
        )
        logger.info(f"AzureBlobLoader initialized for container '{settings.AZURE_STORAGE_CONTAINER_NAME}'.")

    def load(self, data_path: str) -> Generator[Document, None, None]:
        # The 'data_path' argument is ignored here as we load directly from the cloud
        logger.info("Loading documents from Azure Blob Storage...")
        yield from self.loader.load()

class AzureEmbedder(BaseEmbedder):
    """Creates embeddings using an Azure OpenAI deployment."""
    def __init__(self):
        self.client = AzureOpenAIEmbeddings(
            azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_KEY,
        )
        logger.info(f"AzureEmbedder initialized for deployment '{settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT}'.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.client.embed_documents(texts)

    @property
    def vector_size(self) -> int:
        return settings.AZURE_OPENAI_EMBEDDING_DIMENSIONS

class AzureVectorStore(BaseVectorStore):
    """Manages an index in Azure AI Search."""
    def __init__(self, embedder_client: AzureOpenAIEmbeddings):
        self.index_name = settings.AZURE_SEARCH_INDEX_NAME
        # The AzureSearch client handles index creation internally if it doesn't exist
        self.client = AzureSearch(
            azure_search_endpoint=settings.AZURE_SEARCH_ENDPOINT,
            azure_search_key=settings.AZURE_SEARCH_KEY,
            index_name=self.index_name,
            embedding_function=embedder_client.embed_query,
        )
        logger.info(f"AzureVectorStore initialized for index '{self.index_name}'.")

    def setup(self, collection_name: str, vector_size: int):
        # The LangChain AzureSearch integration handles index creation on initialization
        # so this method can be a no-op, but we log to confirm.
        logger.info(f"Azure AI Search index '{self.index_name}' setup is handled by the client constructor.")
        pass

    def upsert(self, documents: List[Document], collection_name: str):
        # The LangChain client takes care of mapping and upserting.
        # We need to extract just the page_content and metadata for the add_documents method.
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        self.client.add_texts(texts=texts, metadatas=metadatas)

class AzureLLM(BaseLLM):
    """Interacts with an Azure OpenAI Chat deployment (e.g., GPT-4)."""
    def __init__(self, prompt_manager: PromptManager):
        self.client = AzureChatOpenAI(
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_KEY,
            temperature=1.0, # For predictable outputs
        )
        self.summary_prompt = prompt_manager.get_prompt("generate_summary")
        self.keyword_prompt = prompt_manager.get_prompt("generate_keywords")
        self.summary_chain = self.summary_prompt | self.client
        self.keyword_chain = self.keyword_prompt | self.client
        logger.info(f"AzureLLM initialized for deployment '{settings.AZURE_OPENAI_CHAT_DEPLOYMENT}'.")

    async def _generate_for_one_text(self, text: str) -> Dict[str, Any]:
        # generating summary and keywords in parallel
        summary_res, keyword_res = await asyncio.gather(
            self.summary_chain.ainvoke({"text": text}),
            self.keyword_chain.ainvoke({"text": text})
        )
        keywords = [k.strip() for k in keyword_res.content.split(",")]
        return {"summary": summary_res.content, "keywords": keywords}

    async def ainvoke_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        tasks = [self._generate_for_one_text(text) for text in texts]
        # await for all the tasks to complete in the batch
        results = await asyncio.gather(*tasks)
        return results
