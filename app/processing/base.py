from abc import ABC, abstractmethod

class PolicyProcessor(ABC):
    @abstractmethod
    def process_policy(self, policy_doc):
        """Process the policy document and return the result."""
        pass

class LLMService(ABC):
    @abstractmethod
    def generate(self, prompt: str, provider: str ) -> str:
        """Generate a response based on the provided prompt."""
        pass