import os
import json
import logging
from typing import List, Dict, Any
from fastapi import HTTPException, status

from app.models.policy_models import SimilarPolicyRequest, PolicyMatch
from app.pipelines.components.vector_store import AzureVectorStore

logger = logging.getLogger(__name__)

class SimilarityService:
    """Service for policy similarity search operations."""
    
    @staticmethod
    async def find_similar_policies(
        request: SimilarPolicyRequest, 
        store: AzureVectorStore
    ) -> List[PolicyMatch]:
        """Find policies similar to the user's description."""
        logger.info(f"Finding similar policies with threshold {request.threshold} for: '{request.description[:50]}...'")
        
        try:
            # Generate embedding for the query
            query_vector = store.embedder.embed_query(request.description)
            
            # Search for similar policies
            results = store.search_similar_policies(
                query_vector=query_vector,
                threshold=request.threshold,
                top_k=5
            )
            
            logger.info(f"Found {len(results)} policies above threshold")
            
            # Process results
            similar_policies = []
            for result in results:
                filename = os.path.basename(result["source"])
                
                # Enhance TOC with topic IDs from the stored topic_ids field
                enhanced_toc = await SimilarityService._enhance_toc_with_ids(
                    result["toc"],
                    result.get("topic_ids", []),
                    store
                )
                
                policy_match = PolicyMatch(
                    policy_id=result["id"],
                    filename=filename,
                    description=result["description"],
                    similarity_score=round(result["similarity_score"] * 100, 2),
                    meets_threshold=True,
                    toc=enhanced_toc,
                    raw_azure_score=result.get("similarity_score", 0),
                    rank_position=result.get("rank", 1),
                    threshold_used=request.threshold,
                    source=result["source"]
                )
                similar_policies.append(policy_match)
                logger.info(f"Added policy: {filename} (similarity: {policy_match.similarity_score}%)")
            
            # Sort by similarity score
            similar_policies.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.info(f"Final result: {len(similar_policies)} policies above {request.threshold*100}% threshold")
            return similar_policies
            
        except Exception as e:
            logger.error(f"Similar policy search failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to find similar policies: {str(e)}"
            )
    
    @staticmethod
    async def _enhance_toc_with_ids(
        toc_structure: List[Dict],
        topic_ids: List[str],
        store: AzureVectorStore
    ) -> List[Dict]:
        """
        Enhance TOC structure with topic and subtopic IDs.
        Simplified version that uses the stored topic_ids from policy document.
        """
        try:
            if not topic_ids:
                logger.warning("No topic IDs provided for TOC enhancement")
                return toc_structure
            
            # Get all topics for this policy
            topics = []
            for topic_id in topic_ids:
                topic = store.get_topic_by_id(topic_id)
                if topic:
                    topics.append(topic)
            
            # Create mapping of topic names to their data
            topic_map = {}
            for topic in topics:
                topic_name = topic.get("topic", "")
                topic_map[topic_name] = {
                    "id": topic.get("id"),
                    "subtopics": topic.get("subtopics", [])
                }
            
            # Enhance TOC recursively
            def enhance_toc_recursive(toc_nodes):
                enhanced_nodes = []
                for node in toc_nodes:
                    if isinstance(node, dict):
                        topic_name = node.get("topic", "")
                        enhanced_node = node.copy()
                        
                        # Add IDs if we found matching topic
                        if topic_name in topic_map:
                            topic_data = topic_map[topic_name]
                            enhanced_node["topic_id"] = topic_data["id"]
                            
                            # Add subtopic IDs
                            if "subtopics" in enhanced_node and enhanced_node["subtopics"]:
                                enhanced_subtopics = []
                                for subtopic_node in enhanced_node["subtopics"]:
                                    subtopic_name = subtopic_node.get("topic", "")
                                    enhanced_subtopic = subtopic_node.copy()
                                    
                                    # Find matching subtopic by name
                                    for stored_subtopic in topic_data["subtopics"]:
                                        if stored_subtopic.get("name") == subtopic_name:
                                            enhanced_subtopic["subtopic_id"] = stored_subtopic.get("id")
                                            break
                                    
                                    enhanced_subtopics.append(enhanced_subtopic)
                                enhanced_node["subtopics"] = enhanced_subtopics
                        else:
                            logger.warning(f"No topic found for TOC item: '{topic_name}'")
                        
                        enhanced_nodes.append(enhanced_node)
                
                return enhanced_nodes
            
            enhanced_toc = enhance_toc_recursive(toc_structure)
            return enhanced_toc
            
        except Exception as e:
            logger.error(f"Failed to enhance TOC: {e}")
            return toc_structure