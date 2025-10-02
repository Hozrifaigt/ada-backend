import logging
import json
from typing import Optional
from fastapi import HTTPException, status

from app.models.policy_models import TopicInfo
from app.pipelines.components.vector_store import AzureVectorStore

logger = logging.getLogger(__name__)

class TopicService:
    """Service for topic management operations."""
    
    @staticmethod
    async def get_topic_by_id(
        topic_id: str, 
        subtopic_id: Optional[str],
        store: AzureVectorStore
    ) -> TopicInfo:
        """
        Get a specific topic by its ID.
        If subtopic_id is provided, returns topic with that specific subtopic.
        Otherwise returns topic with all subtopics.
        """
        logger.info(f"Getting topic with ID: '{topic_id}', subtopic: '{subtopic_id}'")
        
        try:
            # Get topic document from store
            topic_doc = store.get_topic_by_id(topic_id)
            
            if not topic_doc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Topic with ID '{topic_id}' not found"
                )
            
            # Build response based on whether subtopic is requested
            topic_name = topic_doc.get('topic', 'Unknown Topic')
            topic_content = topic_doc.get('topic_content', '')
            subtopics = topic_doc.get('subtopics', [])
            
            # If specific subtopic requested, filter to just that one
            if subtopic_id:
                filtered_subtopics = []
                for subtopic in subtopics:
                    if subtopic.get('id') == subtopic_id:
                        filtered_subtopics = [subtopic]
                        break
                
                if not filtered_subtopics:
                    logger.warning(f"Subtopic '{subtopic_id}' not found in topic '{topic_id}'")
                
                return TopicInfo(
                    topic_id=topic_id,
                    topic_name=topic_name,
                    topic_content=topic_content,
                    subtopics=filtered_subtopics
                )
            else:
                # Return all subtopics
                return TopicInfo(
                    topic_id=topic_id,
                    topic_name=topic_name,
                    topic_content=topic_content,
                    subtopics=subtopics
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve topic: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve topic: {str(e)}"
            )