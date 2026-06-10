import uuid
import json
import logging

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.config.settings import settings
from app.pipelines.components.embedder import AzureEmbedder
from app.models.regulation_models import RegulationChunk

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """Result of document upload operation."""
    successful: int
    failed: int
    total: int
    
    @property
    def success_rate(self) -> float:
        return self.successful / self.total if self.total > 0 else 0.0


class AzureVectorStore:
    """Simplified Azure AI Search store - vectors only for policy descriptions."""

    def __init__(self, embedder: AzureEmbedder):
        self.index_name = settings.AZURE_SEARCH_INDEX_NAME
        self.regulations_index_name = settings.AZURE_SEARCH_REGULATIONS_INDEX_NAME
        self.embedder = embedder
        self.llm_service = None

        logger.info(f"Initializing Simplified AzureVectorStore:")
        logger.info(f"Endpoint: {settings.AZURE_SEARCH_ENDPOINT}")
        logger.info(f"Policies Index: {self.index_name}")
        logger.info(f"Regulations Index: {self.regulations_index_name}")
        
        # Initialize LLM service for synonym expansion
        try:
            from app.pipelines.components.llm import AzureLLM
            from app.prompts.manager import PromptManager
            self.llm_service = AzureLLM(prompt_manager=PromptManager(settings.PROMPT_MANAGER_PATH))
            logger.info("LLM service initialized for synonym expansion")
        except Exception as e:
            logger.warning(f"Could not initialize LLM service for synonym expansion: {e}")

        try:
            # Client for policies and topics
            self.search_client = SearchClient(
                endpoint=settings.AZURE_SEARCH_ENDPOINT,
                index_name=self.index_name,
                credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
            )

            # Separate client for regulations
            self.regulations_client = SearchClient(
                endpoint=settings.AZURE_SEARCH_ENDPOINT,
                index_name=self.regulations_index_name,
                credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
            )
            logger.info("AzureVectorStore initialized with separate regulations index")
        except Exception as e:
            logger.error(f"Failed to initialize AzureVectorStore: {e}")
            raise
    
    async def _expand_synonyms(self, topic: str) -> List[str]:
        """
        Expand a topic with synonyms using LLM for better search coverage.
        
        Args:
            topic: The main topic to expand
            
        Returns:
            List of synonyms and variations (includes original topic)
        """
        synonyms = [topic]  # Always include original
        
        if not self.llm_service:
            logger.debug(f"LLM service unavailable, returning original topic: {topic}")
            return synonyms
        
        try:
            prompt = f"""You are helping expand search queries for UAE labor regulations.

Given the topic: {topic}

Provide 4-5 relevant synonyms or variations that a user might use when searching for this concept.
Include common abbreviations, related terms, and different phrasings.

Return ONLY a JSON array of strings, no other text.
Example format: ["synonym1", "synonym2", "synonym3", "synonym4", "synonym5"]"""
            
            response = await self.llm_service.generate(prompt)
            
            # Parse JSON response
            import json
            expanded = json.loads(response)
            
            if isinstance(expanded, list):
                # Deduplicate and combine with original
                synonyms.extend([s for s in expanded if s.lower() != topic.lower()])
                synonyms = list(dict.fromkeys(synonyms))  # Preserve order, remove duplicates
                logger.debug(f"Expanded '{topic}' to synonyms: {synonyms}")
            
        except Exception as e:
            logger.debug(f"Failed to expand synonyms for '{topic}': {e}. Using original only.")
        
        return synonyms
    
    def _create_base_document(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Create base document structure with required fields."""
        return {
            "id": doc_id or str(uuid.uuid4()),
            "record_type": "",
            # Policy fields
            "filename": "",
            "source": "",
            "description": "",
            "description_vector": [],
            "toc": "[]",
            "topic_ids": [],
            # Topic fields (with vectors)
            "policy_id": "",
            "topic": "",
            "topic_content": "",
            "topic_vector": [],
            "subtopics": "[]"
        }
    
    def _create_policy_document(self, policy_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Create properly structured policy document for Azure Search."""
        search_doc = self._create_base_document(policy_doc.get("id"))
        
        # Serialize TOC to JSON string
        toc_json = json.dumps(policy_doc.get('toc', []))
        topic_ids = policy_doc.get('topic_ids', [])
        
        search_doc.update({
            "record_type": "policy",
            "filename": policy_doc.get('filename', ''),
            "source": policy_doc.get('source', ''),
            "web_url": policy_doc.get('web_url', ''),
            "function": policy_doc.get('function', ''),
            "description": policy_doc.get('description', ''),
            "description_vector": policy_doc.get('description_vector', []),
            "toc": toc_json,
            "topic_ids": topic_ids
        })
        
        return search_doc
    
    def _create_topic_document(self, topic_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Create topic document with vectors for similarity search."""
        search_doc = self._create_base_document(topic_doc.get("id"))

        # Subtopics already come as JSON string from pipeline
        subtopics_json = topic_doc.get('subtopics', '[]')
        if not isinstance(subtopics_json, str):
            subtopics_json = json.dumps(subtopics_json)

        search_doc.update({
            "record_type": "topic",
            "policy_id": topic_doc.get('policy_id', ''),
            "source": topic_doc.get('source', ''),
            "topic": topic_doc.get('topic', ''),
            "topic_content": topic_doc.get('topic_content', ''),
            "topic_vector": topic_doc.get('topic_vector', []),
            "subtopics": subtopics_json
        })

        return search_doc
    
    def _upload_batch(self, documents: List[Dict[str, Any]]) -> UploadResult:
        """Upload a batch of documents and return results."""
        try:
            result = self.search_client.upload_documents(documents)
            
            successful = sum(1 for item in result if item.succeeded)
            failed = len(result) - successful
            
            if failed > 0:
                logger.error(f"Failed to upload {failed} documents in batch")
                for item in result:
                    if not item.succeeded:
                        logger.error(f"Upload error for doc {getattr(item, 'key', 'unknown')}: {item.error_message}")
            
            return UploadResult(successful=successful, failed=failed, total=len(documents))
            
        except AzureError as e:
            logger.error(f"Azure error during batch upload: {e}")
            return UploadResult(successful=0, failed=len(documents), total=len(documents))
        except Exception as e:
            logger.error(f"Unexpected error during batch upload: {e}")
            return UploadResult(successful=0, failed=len(documents), total=len(documents))
    
    def upsert_policy_documents(self, policy_docs: List[Dict[str, Any]]) -> bool:
        """Store policy documents with embeddings."""
        if not policy_docs:
            logger.warning("No policy documents to store")
            return True
        
        logger.info(f"Storing {len(policy_docs)} policy documents (with embeddings)")
        
        try:
            documents_to_upload = []
            for policy_doc in policy_docs:
                search_doc = self._create_policy_document(policy_doc)
                documents_to_upload.append(search_doc)
                logger.debug(f"Prepared policy: {policy_doc.get('filename', 'unknown')}")
            
            result = self._upload_batch(documents_to_upload)

            # web_url was added to the schema later. If the live index doesn't have
            # that column yet, policy uploads fail. Retry without web_url so policies
            # still index (the SharePoint link just won't be available until the
            # index is recreated with the field).
            if result.failed > 0:
                logger.warning(
                    f"{result.failed} policy upload(s) failed — retrying without 'web_url' "
                    f"(index may not have the field; run reset-index to enable links)"
                )
                stripped = []
                for doc in documents_to_upload:
                    doc.pop("web_url", None)
                    stripped.append(doc)
                result = self._upload_batch(stripped)

            logger.info(f"Policy upload complete: {result.successful} successful, {result.failed} failed "
                       f"(success rate: {result.success_rate:.2%})")
            return result.failed == 0

        except Exception as e:
            logger.error(f"Failed to store policy documents: {e}")
            return False
    
    def upsert_topic_documents(self, topic_docs: List[Dict[str, Any]]) -> bool:
        """Store topic documents WITHOUT embeddings - retrieval by ID only."""
        if not topic_docs:
            logger.warning("No topic documents to store")
            return True
        
        logger.info(f"Storing {len(topic_docs)} topic documents (no embeddings)")
        
        try:
            batch_size = getattr(settings, 'BATCH_SIZE', 24)
            total_result = UploadResult(successful=0, failed=0, total=len(topic_docs))
            
            for i in range(0, len(topic_docs), batch_size):
                batch = topic_docs[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(topic_docs) - 1) // batch_size + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                documents_to_upload = []
                for topic_doc in batch:
                    search_doc = self._create_topic_document(topic_doc)
                    documents_to_upload.append(search_doc)
                
                batch_result = self._upload_batch(documents_to_upload)
                
                total_result.successful += batch_result.successful
                total_result.failed += batch_result.failed
                
                logger.info(f"Batch {batch_num} complete: {batch_result.successful} successful, "
                           f"{batch_result.failed} failed")
            
            logger.info(f"Topic upload complete: {total_result.successful} successful, "
                       f"{total_result.failed} failed (success rate: {total_result.success_rate:.2%})")
            return total_result.failed == 0
            
        except Exception as e:
            logger.error(f"Failed to store topic documents: {e}")
            return False
    
    def search_similar_policies(self, query_vector: List[float], threshold: float = 0.6, top_k: int = 5,
                                function: Optional[str] = None) -> List[Dict]:
        """
        Search for similar policies using vector similarity on descriptions.
        This is the ONLY vector search in the system.
        If `function` is provided, only policies in that same function are considered
        (e.g. an HR draft is never matched against an IT policy).
        """
        try:
            from azure.search.documents.models import VectorizedQuery

            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="description_vector"
            )

            # Only compare against policies of the same function (categorical guard).
            filter_expr = "record_type eq 'policy'"
            if function:
                safe_function = function.strip().lower().replace("'", "''")
                filter_expr += f" and function eq '{safe_function}'"
            logger.info(f"Policy search filter: {filter_expr}")

            # web_url was added to the schema later; older indexes may not have it.
            # Try selecting it, and fall back to a select without it if Azure rejects it.
            base_select = "id,filename,description,source,toc,topic_ids"
            try:
                results = list(self.search_client.search(
                    search_text="",
                    vector_queries=[vector_query],
                    filter=filter_expr,
                    select=base_select + ",web_url",
                    top=top_k
                ))
            except Exception as select_err:
                logger.warning(f"web_url not in index, retrying without it: {select_err}")
                results = list(self.search_client.search(
                    search_text="",
                    vector_queries=[vector_query],
                    filter=filter_expr,
                    select=base_select,
                    top=top_k
                ))

            processed_results = []
            for i, result in enumerate(results):
                toc = json.loads(result.get("toc", "[]"))
                # Azure @search.score for cosine = 1 / (2 - cosine_similarity).
                # Invert it back to the true cosine similarity so the threshold
                # is intuitive and meaningful for text-embedding-ada-002 vectors.
                azure_score = result.get('@search.score', 0.0)
                cosine_similarity = 2 - (1 / azure_score) if azure_score > 0 else -1.0
                logger.info(
                    f"Policy similarity: '{result.get('filename', '?')}' "
                    f"cosine={cosine_similarity:.4f} (azure={azure_score:.4f}, threshold={threshold})"
                )

                if cosine_similarity >= threshold:
                    processed_results.append({
                        "id": result["id"],
                        "filename": result["filename"],
                        "description": result["description"],
                        "source": result["source"],
                        "web_url": result.get("web_url", ""),
                        "toc": toc,
                        "topic_ids": result.get("topic_ids", []),
                        "similarity_score": cosine_similarity,
                        "rank": i + 1
                    })

            logger.info(f"Policies above cosine threshold {threshold}: {len(processed_results)}/{top_k}")
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to search similar policies: {e}")
            return []

    def search_similar_topics(self, query_vector: List[float], threshold: float = 0.6, top_k: int = 5) -> List[Dict]:
        """
        Search for similar topics using vector similarity on topic content.
        """
        try:
            from azure.search.documents.models import VectorizedQuery

            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="topic_vector"
            )

            results = self.search_client.search(
                search_text="",
                vector_queries=[vector_query],
                filter="record_type eq 'topic'",
                select="id,policy_id,topic,topic_content,subtopics,source",
                top=top_k
            )

            processed_results = []
            for i, result in enumerate(results):
                # Parse subtopics JSON
                subtopics = json.loads(result.get("subtopics", "[]"))

                # Get the actual similarity score from Azure Search
                # Azure returns scores as @search.score, which is accessible as a property
                similarity_score = result.get('@search.score', 0.0)

                if similarity_score >= threshold:
                    processed_results.append({
                        "id": result["id"],
                        "policy_id": result["policy_id"],
                        "topic": result["topic"],
                        "topic_content": result["topic_content"],
                        "subtopics": subtopics,
                        "source": result["source"],
                        "similarity_score": similarity_score,
                        "rank": i + 1
                    })

            return processed_results

        except Exception as e:
            logger.error(f"Failed to search similar topics: {e}")
            return []

    def get_topic_by_id(self, topic_id: str) -> Optional[Dict]:
        """
        Get a topic document by its ID - direct lookup, no vector search.
        """
        try:
            topic = self.search_client.get_document(key=topic_id)
            
            if topic and topic.get("record_type") == "topic":
                # Parse subtopics JSON
                topic["subtopics"] = json.loads(topic.get("subtopics", "[]"))
                return topic
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get topic {topic_id}: {e}")
            
            # Fallback to search
            try:
                results = list(self.search_client.search(
                    search_text="*",
                    filter=f"record_type eq 'topic' and id eq '{topic_id}'",
                    select="id,policy_id,topic,topic_content,subtopics,source",
                    top=1
                ))
                
                if results:
                    topic = results[0]
                    topic["subtopics"] = json.loads(topic.get("subtopics", "[]"))
                    return topic
                
                return None
                
            except Exception as e:
                logger.error(f"Failed to retrieve topic by search: {e}")
                return None
    
    def get_topics_by_policy(self, policy_id: str) -> List[Dict]:
        """
        Get all topics for a specific policy using the stored topic_ids.
        Direct ID lookups only - no vector search.
        """
        try:
            # First get the policy document to retrieve topic IDs
            policy = self.search_client.get_document(key=policy_id)
            
            if not policy or policy.get("record_type") != "policy":
                logger.warning(f"Policy {policy_id} not found")
                return []
            
            topic_ids = policy.get("topic_ids", [])
            
            if not topic_ids:
                logger.warning(f"No topics found for policy {policy_id}")
                return []
            
            # Retrieve all topics by their IDs
            topics = []
            for topic_id in topic_ids:
                topic = self.get_topic_by_id(topic_id)
                if topic:
                    topics.append(topic)
                else:
                    logger.warning(f"Topic {topic_id} not found for policy {policy_id}")
            
            return topics
            
        except Exception as e:
            logger.error(f"Failed to get topics for policy {policy_id}: {e}")
            return []
    
    def get_policy_by_id(self, policy_id: str) -> Optional[Dict]:
        """
        Get a policy document by its ID.
        """
        try:
            policy = self.search_client.get_document(key=policy_id)
            
            if policy and policy.get("record_type") == "policy":
                # Parse TOC JSON
                policy["toc"] = json.loads(policy.get("toc", "[]"))
                return policy
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get policy {policy_id}: {e}")
            return None
    
    def get_all_policies(self, max_results: int = 50) -> List[Dict]:
        """
        Get all policy documents (no vector search, just listing).
        """
        try:
            results = self.search_client.search(
                search_text="*",
                filter="record_type eq 'policy'",
                select="id,filename,description,source,toc,topic_ids",
                top=max_results
            )

            policies = []
            for result in results:
                # Parse TOC JSON
                result["toc"] = json.loads(result.get("toc", "[]"))
                policies.append(result)

            return policies

        except Exception as e:
            logger.error(f"Failed to retrieve all policies: {e}")
            return []

    def delete_policy_by_source(self, source: str) -> bool:
        """
        Delete a policy and all its topics by the source identifier.
        Used by the SharePoint CDC pipeline when a file is removed from the library.
        """
        try:
            results = list(self.search_client.search(
                search_text="*",
                filter=f"record_type eq 'policy' and source eq '{source}'",
                select="id,topic_ids",
                top=10
            ))
            if not results:
                logger.info(f"No policy found with source '{source}', nothing to delete")
                return True

            for policy in results:
                topic_ids = policy.get("topic_ids", [])
                if topic_ids:
                    topic_delete_batch = [{"id": tid} for tid in topic_ids]
                    self.search_client.delete_documents(documents=topic_delete_batch)
                    logger.info(f"Deleted {len(topic_ids)} topics for policy {policy['id']}")
                self.search_client.delete_documents(documents=[{"id": policy["id"]}])
                logger.info(f"Deleted policy {policy['id']} (source: {source})")
            return True
        except Exception as e:
            logger.error(f"Failed to delete policy by source '{source}': {e}")
            return False

    # ============================================================
    # REGULATIONS METHODS
    # ============================================================

    def _create_regulation_document(self, regulation_chunk: RegulationChunk) -> Dict[str, Any]:
        """Create Azure Search document from RegulationChunk."""
        return {
            "id": regulation_chunk.id,
            "record_type": "regulation",
            "content": regulation_chunk.content,
            "regulation_category": regulation_chunk.regulation_category,
            "decree_name": regulation_chunk.decree_name,
            "article_number": regulation_chunk.article_number or "",
            "regulation_type": regulation_chunk.regulation_type,
            "effective_date": regulation_chunk.effective_date or "",
            "topics": regulation_chunk.topics,
            # Vector will be added during indexing
            "content_vector": []
        }

    def _upload_regulations_batch(self, documents: List[Dict[str, Any]]) -> UploadResult:
        """Upload a batch of regulation documents using regulations client."""
        try:
            result = self.regulations_client.upload_documents(documents)

            successful = sum(1 for item in result if item.succeeded)
            failed = len(result) - successful

            if failed > 0:
                logger.error(f"Failed to upload {failed} regulation documents in batch")
                for item in result:
                    if not item.succeeded:
                        logger.error(f"Upload error for doc {getattr(item, 'key', 'unknown')}: {item.error_message}")

            return UploadResult(successful=successful, failed=failed, total=len(documents))

        except AzureError as e:
            logger.error(f"Azure error during regulation batch upload: {e}")
            return UploadResult(successful=0, failed=len(documents), total=len(documents))
        except Exception as e:
            logger.error(f"Unexpected error during regulation batch upload: {e}")
            return UploadResult(successful=0, failed=len(documents), total=len(documents))

    async def index_regulations(self, chunks: List[RegulationChunk]) -> bool:
        """
        Index regulation chunks with embeddings in Azure Search.

        Args:
            chunks: List of RegulationChunk objects to index

        Returns:
            True if successful, False otherwise
        """
        if not chunks:
            logger.warning("No regulation chunks to index")
            return True

        logger.info(f"Indexing {len(chunks)} regulation chunks into '{self.regulations_index_name}'")

        try:
            documents_to_upload = []

            # Build enriched texts for embedding that include topics and metadata
            # This creates powerful vectors that understand the semantic context
            enriched_texts = []
            for chunk in chunks:
                # Concatenate topics + decree_name + article_number + content
                # This way the embedding captures all semantic signals
                topics_str = " ".join(chunk.topics) if chunk.topics else ""
                article_ref = f"{chunk.article_number}" if chunk.article_number else ""
                
                enriched_text = f"{topics_str} | {chunk.decree_name} | {article_ref} | {chunk.content}"
                enriched_texts.append(enriched_text)

            # Generate embeddings in batch for better performance
            logger.info(f"Generating enriched embeddings for {len(enriched_texts)} chunks (with topics + metadata)...")
            embeddings = self.embedder.embed_documents(enriched_texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings")

            # Create search documents with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                search_doc = self._create_regulation_document(chunk)
                search_doc["content_vector"] = embedding
                documents_to_upload.append(search_doc)

            # Upload in batches using regulations client
            batch_size = getattr(settings, 'BATCH_SIZE', 24)
            total_result = UploadResult(successful=0, failed=0, total=len(documents_to_upload))

            for i in range(0, len(documents_to_upload), batch_size):
                batch = documents_to_upload[i:i + batch_size]
                batch_result = self._upload_regulations_batch(batch)

                total_result.successful += batch_result.successful
                total_result.failed += batch_result.failed

                logger.info(f"Batch {i // batch_size + 1} complete: {batch_result.successful} successful")

            logger.info(f"Regulation indexing complete: {total_result.successful} successful, "
                       f"{total_result.failed} failed")

            return total_result.failed == 0

        except Exception as e:
            logger.error(f"Failed to index regulations: {e}")
            return False

    async def search_relevant_regulations(
        self,
        query_text: str,
        regulation_name: Optional[str] = None,
        top_k: int = None,
        min_score: float = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[RegulationChunk]:
        """
        Search for relevant regulation chunks using simple keyword search.

        Args:
            query_text: The search query
            regulation_name: Optional filter by regulation category
            top_k: Number of results to return
            min_score: Minimum similarity score threshold (not used, kept for compatibility)
            context: Additional context (not used, kept for compatibility)

        Returns:
            List of RegulationChunk objects with relevance scores
        """
        if top_k is None:
            top_k = settings.REGULATION_SEARCH_TOP_K

        try:
            logger.info(f"Searching regulations for: {query_text}")

            # Generate embedding for semantic search
            query_vector = self.embedder.embed_query(query_text)

            from azure.search.documents.models import VectorizedQuery

            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )

            # Hybrid search: BM25 (keywords) + Vector (semantic)
            results = self.regulations_client.search(
                search_text=query_text,
                search_fields=["topics", "content"],  # BM25 searches these
                vector_queries=[vector_query],  # Vector searches content_vector
                select="id,content,regulation_category,decree_name,article_number,regulation_type,effective_date,topics",
                top=top_k
            )

            # Process results - Azure combines BM25 + vector scores
            processed_results = []
            for result in results:
                score = result.get('@search.score', 0.0)

                chunk = RegulationChunk(
                    id=result["id"],
                    content=result["content"],
                    regulation_category=result["regulation_category"],
                    decree_name=result["decree_name"],
                    article_number=result.get("article_number", ""),
                    regulation_type=result["regulation_type"],
                    effective_date=result.get("effective_date"),
                    topics=result.get("topics", []),
                    score=score
                )
                processed_results.append(chunk)

            logger.info(f"Found {len(processed_results)} regulation chunks (hybrid BM25 + vector)")

            return processed_results

        except Exception as e:
            logger.error(f"Failed to search regulations: {e}")
            return []

    async def search_regulations_by_article(
        self,
        article_query: str,
        regulation_name: Optional[str] = None
    ) -> List[RegulationChunk]:
        """
        Search for specific articles by number or topic.

        This is used when the user requests a specific article.

        Args:
            article_query: Article number or topic (e.g., "Article 15" or "leave entitlements")
            regulation_name: Optional regulation name to filter by

        Returns:
            List of matching RegulationChunk objects
        """
        try:
            filter_parts = ["record_type eq 'regulation'"]

            if regulation_name:
                filter_parts.append(f"regulation_category eq '{regulation_name}'")

            filter_query = " and ".join(filter_parts)

            # Use keyword search for article numbers
            results = self.regulations_client.search(
                search_text=article_query,
                filter=filter_query,
                select="id,content,regulation_category,decree_name,article_number,regulation_type,effective_date,topics",
                top=10
            )

            chunks = []
            for result in results:
                chunk = RegulationChunk(
                    id=result["id"],
                    content=result["content"],
                    regulation_category=result["regulation_category"],
                    decree_name=result["decree_name"],
                    article_number=result.get("article_number", ""),
                    regulation_type=result["regulation_type"],
                    effective_date=result.get("effective_date"),
                    topics=result.get("topics", [])
                )
                chunks.append(chunk)

            logger.info(f"Found {len(chunks)} articles matching '{article_query}'")
            return chunks

        except Exception as e:
            logger.error(f"Failed to search regulations by article: {e}")
            return []