import logging
import asyncio

from pydantic import BaseModel, Field
from typing import List, Generator, Dict, Any

from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_community.document_loaders import AzureBlobStorageContainerLoader



from app.config.settings import settings
from app.prompts.manager import PromptManager 

logger = logging.getLogger(__name__)


class AzureBlobLoader:
    """Loads .docx files from an Azure Blob Storage container."""


    def __init__(self):
        if not settings.AZURE_STORAGE_CONNECTION_STRING or not settings.AZURE_STORAGE_CONTAINER_NAME:
            raise ValueError("Azure Storage settings are not configured.")
        self.loader = AzureBlobStorageContainerLoader(...)

    def load(self, data_path: str) -> Generator[Document, None, None]:
        yield from self.loader.load()

class AzureEmbedder:
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

class AzureVectorStore:
    def __init__(self, embedder_client: AzureOpenAIEmbeddings):
        self.index_name = settings.AZURE_SEARCH_INDEX_NAME
        embedding_function = embedder_client.embed_query

        self.client = AzureSearch(
            azure_search_endpoint=settings.AZURE_SEARCH_ENDPOINT,
            azure_search_key=settings.AZURE_SEARCH_KEY,
            index_name=self.index_name,
            embedding_function=embedding_function,
        )
        logger.info(f"AzureVectorStore initialized for index '{self.index_name}'.")

    def upsert(self, documents: List[Document], collection_name: str):
        """
        Upserts fully processed documents into Azure AI Search.
        It assumes EVERY document has a pre-computed vector in its metadata.
        """
        if not documents:
            logger.warning("Upsert called with no documents to process.")
            return

        logger.info(f"Upserting {len(documents)} documents with pre-computed vectors to '{collection_name}'.")

        # The pipeline guarantees all documents have embeddings.
        # The add_texts method automatically finds the 'embedding' key in the metadata.
        self.client.add_documents(documents=documents)
        logger.info("Upsert operation completed successfully.")



class AzureLLM:
    """
    Interacts with Azure OpenAI. Its primary complex task is to perform
    Map-Reduce summarization for large policy documents.
    """
    def __init__(self, prompt_manager: PromptManager):
        self.client = AzureChatOpenAI(
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_KEY,
            temperature=0.3,
        )
        
        # This is now the ONLY complex chain this class manages.
        self.map_reduce_summarize_chain = load_summarize_chain(
            llm=self.client, 
            chain_type="map_reduce"
        )

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=400)
        logger.info(f"AzureLLM initialized for document summarization tasks.")

    async def asummarize_document(self, document: Document) -> str:
        """
        Generates a comprehensive summary for a large document using Map-Reduce.
        Returns the summary text directly.
        """
        logger.info(f"Starting Map-Reduce summarization for: {document.metadata.get('filename')}")
        
        docs_for_summarization = self.text_splitter.create_documents([document.page_content])
        
        try:
            result = await self.map_reduce_summarize_chain.ainvoke(docs_for_summarization)
            return result.get('output_text', '')
        except Exception as e:
            logger.error(f"Map-Reduce summarization failed for {document.metadata.get('filename')}: {e}")
            return "Error: Could not generate a summary for this document."

