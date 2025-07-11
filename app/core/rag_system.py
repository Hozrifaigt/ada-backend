from abc import ABC, abstractmethod
from typing import IO

from app.models.models import (
    PolicyTopicRequest, PolicyTopicResponse, 
    PolicyDraftRequest, PolicyDraftRequest
)

class RAGSystem(ABC):
    """
    Abstract Base Class for a Retrieval-Augmented Generation system.
    Defines the contract for all RAG implementations (local, cloud, etc.).
    """

    @abstractmethod
    def ingest_document(self, file: IO, filename: str) -> None:
        """
        Processes and stores a document in the vector database.
        Accepts a file-like object.
        """
        pass

    @abstractmethod
    def generate_draft(self, request: PolicyDraftRequest) -> PolicyDraftRequest:
        """
        Generates a new policy draft based on a query and existing documents.
        """
        pass
