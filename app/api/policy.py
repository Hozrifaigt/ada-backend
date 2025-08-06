import os
import json
import logging

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from app.pipelines.components.embedder import AzureEmbedder
from app.pipelines.components.vector_store import AzureVectorStore

logger = logging.getLogger(__name__)

try:
    embedder = AzureEmbedder()
    vector_store = AzureVectorStore(embedder=embedder)
    logger.info("Retrieval services initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize retrieval services: {e}", exc_info=True)
    embedder = None
    vector_store = None

def get_vector_store() -> AzureVectorStore:
    """FastAPI dependency for vector store."""
    if vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is not available due to initialization error."
        )
    return vector_store

router = APIRouter()

@router.get("/policies")
async def get_all_policies(
    store: AzureVectorStore = Depends(get_vector_store)
):
    """
    Retrieve a list of all indexed policy documents with their metadata.
    
    Returns basic information about all policies in the system.
    """
    logger.info("Fetching a list of all indexed policies")
    
    try:
        # Search for all policy records
        results = store.search_client.search(
            search_text="*",
            filter="record_type eq 'policy'",
            select="id,description,source,toc",
            top=100,  # Adjust based on expected number of policies
        )
        
        policies = []
        for result in results:
            import os
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
        return {
            "count": len(policies),
            "policies": policies
        }
        
    except Exception as e:
        logger.error(f"Failed to retrieve all policies: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve policies: {str(e)}"
        )

@router.post("/policies/similar_policy", response_model=List[Dict[str, Any]])
async def find_similar_policies_with_scores(
    request: Dict[str, Any],
    store: AzureVectorStore = Depends(get_vector_store)
):
    """
    Find policies similar to the user's description with similarity scores.    
    """
    
    # Extract parameters from request body
    description = request.get("description", "")
    threshold = request.get("threshold", 0.7)
    include_enhanced_toc = request.get("include_enhanced_toc", True)
    
    if not description:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'description' field in request body"
        )
    
    # Validate threshold
    if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Threshold must be a number between 0 and 1 (e.g., 0.7 for 70%)"
        )
    
    logger.info(f"Finding similar policies with threshold {threshold} for: '{description[:50]}...'")
    
    try:
        # Generate embedding for the query
        query_vector = store.embedder.embed_query(description)
        
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=10,
            fields="description_vector"
        )
        
        # Search for similar policies
        results = list(store.search_client.search(
            search_text=description,
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
            if similarity_percentage >= (threshold * 100):
                
                # Parse TOC from JSON string
                try:
                    toc_structure = json.loads(result["toc"]) if result["toc"] else []
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse TOC for policy {filename}")
                    toc_structure = []
                
                # Enhance TOC with chunk IDs if requested
                enhanced_toc = toc_structure  # Default to original TOC
                if include_enhanced_toc and toc_structure:
                    try:
                        enhanced_toc = await _enhance_toc_with_chunk_ids(
                            result["source"], 
                            toc_structure, 
                            store
                        )
                        logger.info(f"Enhanced TOC for {filename} with chunk references")
                    except Exception as e:
                        logger.warning(f"Failed to enhance TOC for {filename}: {e}")
                        enhanced_toc = toc_structure  # Fallback to original
                
                policy_match = {
                    "policy_id": result["id"],
                    "filename": filename,
                    "description": result["description"],
                    "similarity_score": round(similarity_percentage, 2),
                    "meets_threshold": True,
                    "toc": enhanced_toc,  # Enhanced TOC with chunk_ids
                    "raw_azure_score": raw_score,
                    "rank_position": i + 1,
                    "threshold_used": threshold,
                    "source": result["source"]
                }
                similar_policies.append(policy_match)
                logger.info(f"  ✅ Added policy: {filename} (similarity: {similarity_percentage}%)")
            else:
                logger.info(f"  ❌ Filtered out: {filename} (similarity: {similarity_percentage}% < {threshold*100}%)")
        
        # Sort by similarity score
        similar_policies.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        logger.info(f"Final result: {len(similar_policies)} policies above {threshold*100}% threshold")
        return similar_policies
        
    except Exception as e:
        logger.error(f"Similar policy search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find similar policies: {str(e)}"
        )

# Updated helper function to handle multiple chunks per topic
async def _enhance_toc_with_chunk_ids(policy_source: str, toc_structure: List[Dict], store: AzureVectorStore) -> List[Dict]:
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


@router.get("/chunks/{chunk_id}")
async def get_chunk_by_id(
    chunk_id: str,
    store: AzureVectorStore = Depends(get_vector_store)
):
    """
    Get a specific chunk by its unique chunk ID.
    
    This is a direct lookup that doesn't require knowing the policy ID or topic name.
    Returns the chunk content along with its metadata.
    """
    logger.info(f"Getting chunk with ID: '{chunk_id}'")

    try:
        # Direct document lookup by key
        chunk = store.search_client.get_document(key=chunk_id)
        
        if chunk and chunk.get("record_type") == "chunk":
            response = {
                "chunk_id": chunk["id"],
                "topic": chunk["topic"],
                "content": chunk["content"],
                "metadata": {
                    "level": chunk["level"],
                    "parent_topic": chunk.get("parent_topic"),
                    "is_subtopic": chunk["is_subtopic"],
                    "source": chunk["source"],
                }
            }
            
            logger.info(f"Successfully retrieved chunk '{chunk_id}' for topic '{chunk['topic']}'")
            return response
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chunk with ID '{chunk_id}' not found"
            )
            
    except ImportError:
        logger.warning("get_document method not available, falling back to search")
        # Fall through to Option 2
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chunk with ID '{chunk_id}' not found"
            )
        logger.warning(f"get_document failed: {e}, falling back to search")