from abc import ABC, abstractmethod
from typing import Dict, IO, Any

class PolicyProcessor(ABC):
    """
    Abstract Base Class for all document processors.
    It defines a standard interface for processing documents from a stream.
    """
    @abstractmethod
    def process(self, file_stream: IO[bytes]) -> Dict[str, Any]:
        """
        Process a policy document provided as a byte stream.
        
        Args:
            file_stream: A file-like object (stream) containing the document's bytes.

        Returns:
            A dictionary containing the structured TOC and content.
        """
        pass



class LLMService(ABC):
    @abstractmethod
    def generate(self, prompt: str, provider: str ) -> str:
        """Generate a response based on the provided prompt."""
        pass