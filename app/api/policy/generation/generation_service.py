import logging


from app.models.policy_models import PolicyGenerationRequest, PolicyGenerationResponse
from app.pipelines.components.vector_store import AzureVectorStore
from app.pipelines.components.llm import AzureLLM
from .content_generator import ContentGenerator

logger = logging.getLogger(__name__)

class GenerationService:
    """Main service for policy generation operations."""
    
    def __init__(self):
        self.content_generator = ContentGenerator()
    
    async def generate_content(
        self,
        request: PolicyGenerationRequest,
        store: AzureVectorStore,
        llm_service: AzureLLM
    ) -> PolicyGenerationResponse:
        """Main entry point for policy content generation."""
        return await self.content_generator.generate_policy_content(
            request, store, llm_service
        )
