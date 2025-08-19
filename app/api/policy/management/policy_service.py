import os
import json
import logging

from fastapi import HTTPException, status

from app.models.policy_models import PoliciesListResponse
from app.pipelines.components.vector_store import AzureVectorStore

logger = logging.getLogger(__name__)

class PolicyService:
    """Service for basic policy CRUD operations."""
    
    @staticmethod
    async def get_all_policies(store: AzureVectorStore) -> PoliciesListResponse:
        """Retrieve a list of all indexed policy documents with their metadata."""
        logger.info("Fetching a list of all indexed policies")
        
        try:
            # Search for all policy records
            results = store.search_client.search(
                search_text="*",
                filter="record_type eq 'policy'",
                select="id,description,source,toc",
                top=50,
            )
            
            policies = []
            for result in results:
                filename = os.path.basename(result["source"])
                
                # Parse TOC from JSON string
                try:
                    toc_structure = json.loads(result["toc"]) if result["toc"] else []
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse TOC for policy {filename}")
                    toc_structure = []
                
                policy_info = {
                    "policy_id": result["id"],
                    "filename": filename,
                    "source": result["source"],
                    "description": result["description"],
                    "topic_count": len(toc_structure),
                    "toc": toc_structure
                }
                policies.append(policy_info)
            
            logger.info(f"Retrieved {len(policies)} policies")
            return PoliciesListResponse(
                count=len(policies),
                policies=policies
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve all policies: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve policies: {str(e)}"
            )
