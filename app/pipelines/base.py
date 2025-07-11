from abc import ABC, abstractmethod
from typing import List, Generator, Dict, Any
from langchain_core.documents import Document

class BaseLoader(ABC):
    """Contract for loading documents from a source."""
    @abstractmethod
    def load(self, data_path: str) -> Generator[Document, None, None]:
        pass

class BaseEmbedder(ABC):
    """Contract for creating vector embeddings."""
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @property
    @abstractmethod
    def vector_size(self) -> int:
        pass

class BaseVectorStore(ABC):
    """Contract for interacting with a vector database."""
    @abstractmethod
    def setup(self, collection_name: str, vector_size: int):
        pass

    @abstractmethod
    def upsert(self, documents: List[Document], collection_name: str):
        pass

class BaseLLM(ABC):
    """Contract for interacting with a Language Model."""
    @abstractmethod
    async def ainvoke_batch(
        self, texts: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously generates metadata (summary, keywords) for a batch of texts.
        This is the primary method the pipeline will use.
        
        Returns a list of dictionaries, e.g.,
        [{'summary': '...', 'keywords': ['a', 'b']}, {'summary': '...', ...}]
        """
        pass