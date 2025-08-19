import os
import json
import logging
from typing import List, Dict, Any
from fastapi import HTTPException, status
from azure.search.documents.models import VectorizedQuery

from app.models.policy_models import SimilarPolicyRequest, PolicyMatch
from app.pipelines.components.vector_store import AzureVectorStore

logger = logging.getLogger(__name__)

class SimilarityService:
    """Service for policy similarity and search operations."""
    
    @staticmethod
    async def enhance_toc_with_chunk_ids(
        policy_source: str, 
        toc_structure: List[Dict], 
        store: AzureVectorStore
    ) -> List[Dict]:
        """
        Helper function to enhance TOC structure with chunk IDs.
        NOW SUPPORTS: Multiple chunks per topic (returns list of chunk_ids)
        """
        try:
            # Get all chunks for this policy
            chunk_results = list(store.search_client.search(
                search_text="*",
                filter=f"record_type eq 'chunk' and source eq '{policy_source}'",
                select="id,topic,level,parent_topic,is_subtopic",
                top=300
            ))
            
            # Create topic name to chunk IDs mapping (CHANGED: now supports multiple chunks)
            topic_to_chunks = {}
            for chunk in chunk_results:
                topic_name = chunk["topic"]
                
                if topic_name not in topic_to_chunks:
                    topic_to_chunks[topic_name] = []
                
                # Add chunk info to the list for this topic
                chunk_info = {
                    "chunk_id": chunk["id"],
                    "level": chunk["level"],
                    "parent_topic": chunk.get("parent_topic"),
                    "is_subtopic": chunk["is_subtopic"]
                }
                topic_to_chunks[topic_name].append(chunk_info)
            
            # Sort chunks within each topic (by level, then by chunk_id for consistency)
            for topic_name in topic_to_chunks:
                topic_to_chunks[topic_name].sort(key=lambda x: (x["level"], x["chunk_id"]))
            
            # Enhance TOC with chunk IDs
            def enhance_toc_recursive(toc_nodes):
                enhanced_nodes = []
                for node in toc_nodes:
                    if isinstance(node, dict):
                        topic_name = node.get("topic", "")
                        enhanced_node = node.copy()
                        
                        # Add chunk references if we found matching chunks
                        if topic_name in topic_to_chunks:
                            chunk_infos = topic_to_chunks[topic_name]
                            
                            # NEW: chunk_ids is now a LIST of chunk IDs
                            enhanced_node["chunk_ids"] = [chunk["chunk_id"] for chunk in chunk_infos]
                            enhanced_node["chunk_count"] = len(chunk_infos)
                            
                            # For backward compatibility, also include first chunk_id as single field
                            enhanced_node["chunk_id"] = chunk_infos[0]["chunk_id"] if chunk_infos else None
                            
                            # Add metadata from first chunk (for display purposes)
                            if chunk_infos:
                                first_chunk = chunk_infos[0]
                                enhanced_node["chunk_level"] = first_chunk["level"]
                                enhanced_node["is_subtopic"] = first_chunk["is_subtopic"]
                            
                            logger.debug(f"Topic '{topic_name}' has {len(chunk_infos)} chunks")
                        else:
                            enhanced_node["chunk_ids"] = []
                            enhanced_node["chunk_count"] = 0
                            enhanced_node["chunk_id"] = None
                            enhanced_node["chunk_level"] = None
                            enhanced_node["is_subtopic"] = False
                            logger.warning(f"No chunks found for TOC topic: '{topic_name}'")
                        
                        # Recursively enhance subtopics
                        if "subtopics" in enhanced_node and enhanced_node["subtopics"]:
                            enhanced_node["subtopics"] = enhance_toc_recursive(enhanced_node["subtopics"])
                        
                        enhanced_nodes.append(enhanced_node)
                
                return enhanced_nodes
            
            enhanced_toc = enhance_toc_recursive(toc_structure)
            
            # Log summary
            total_topics = len(topic_to_chunks)
            multi_chunk_topics = len([t for t, chunks in topic_to_chunks.items() if len(chunks) > 1])
            if multi_chunk_topics > 0:
                logger.info(f"Enhanced TOC: {total_topics} topics, {multi_chunk_topics} topics have multiple chunks")
            
            return enhanced_toc
            
        except Exception as e:
            logger.error(f"Failed to enhance TOC: {e}")
            return toc_structure

    @staticmethod
    async def find_similar_policies(
        request: SimilarPolicyRequest, 
        store: AzureVectorStore
    ) -> List[PolicyMatch]:
        """Find policies similar to the user's description with similarity scores."""
        logger.info(f"Finding similar policies with threshold {request.threshold} for: '{request.description[:50]}...'")
        
        try:
            # Generate embedding for the query
            query_vector = store.embedder.embed_query(request.description)
            
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=10,
                fields="description_vector"
            )
            
            # Search for similar policies
            results = list(store.search_client.search(
                search_text=request.description,
                vector_queries=[vector_query],
                filter="record_type eq 'policy'",
                select="id,description,source,toc",
                top=5
            ))
            
            logger.info(f"Found {len(results)} total results from search")
            
            # Process results with rank-based scoring
            similar_policies = []
            for i, result in enumerate(results):
                filename = os.path.basename(result["source"])
                
                raw_score = result["@search.score"]
                similarity_percentage = max(95 - (i * 10), 10)  # Rank-based scoring
                
                logger.info(f"Result {i+1}: {filename} - similarity={similarity_percentage}%")
                
                # Apply threshold
                if similarity_percentage >= (request.threshold * 100):
                    
                    # Parse TOC from JSON string
                    try:
                        toc_structure = json.loads(result["toc"]) if result["toc"] else []
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse TOC for policy {filename}")
                        toc_structure = []
                    
                    # Always enhance TOC with chunk IDs
                    enhanced_toc = toc_structure  # Default to original TOC
                    if toc_structure:
                        try:
                            enhanced_toc = await SimilarityService.enhance_toc_with_chunk_ids(
                                result["source"], 
                                toc_structure, 
                                store
                            )
                            logger.info(f"Enhanced TOC for {filename} with chunk references")
                        except Exception as e:
                            logger.warning(f"Failed to enhance TOC for {filename}: {e}")
                            enhanced_toc = toc_structure  # Fallback to original
                    
                    policy_match = PolicyMatch(
                        policy_id=result["id"],
                        filename=filename,
                        description=result["description"],
                        similarity_score=round(similarity_percentage, 2),
                        meets_threshold=True,
                        toc=enhanced_toc,
                        raw_azure_score=raw_score,
                        rank_position=i + 1,
                        threshold_used=request.threshold,
                        source=result["source"]
                    )
                    similar_policies.append(policy_match)
                    logger.info(f"Added policy: {filename} (similarity: {similarity_percentage}%)")
                else:
                    logger.info(f"Filtered out: {filename} (similarity: {similarity_percentage}% < {request.threshold*100}%)")

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
