import logging
import asyncio
import time

from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIConnectionError, APITimeoutError

from app.config.settings import settings
from app.prompts.manager import PromptManager 

logger = logging.getLogger(__name__)

class AzureLLM:
    """Production-ready Azure OpenAI LLM for document processing."""
    
    def __init__(self, prompt_manager: PromptManager):
        self.client = AzureChatOpenAI(
            openai_api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_KEY,
            # request_timeout=120,  # 2 minutes timeout
            # max_retries=5,  # Increase retry attempts
        )
        
        self.prompt_manager = prompt_manager
        self.summarize_prompt = prompt_manager.get_prompt_template("summarize_document")
        self.chunk_size = settings.LLM_CHUNK_SIZE

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=settings.LLM_CHUNK_OVERLAP
        )

        
        logger.info("Production AzureLLM initialized with custom prompts")

    async def generate(self, prompt: str) -> str:
        """Generate content based on a text prompt with retry logic."""
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries + 1):
            try:
                # Create a simple user message from the prompt
                messages = [{"role": "user", "content": prompt}]
                result = await self.client.ainvoke(messages)

                generated_content = result.content.strip() if result.content else "No content generated"
                logger.debug(f"Generated content: {len(generated_content)} characters")
                return generated_content

            except (APIConnectionError, APITimeoutError) as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Connection/timeout error (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Content generation failed after {max_retries + 1} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"Content generation failed: {e}")
                raise

    async def asummarize_document(self, text: str) -> str:
        """Generate document summary using custom prompt with recursive chunking."""
        if not text or not text.strip():
            return "No content to summarize"

        try:
            if len(text.split()) <= self.chunk_size:
                return await self._summarize_text(text)
            return await self._recursive_summarize(text, self.chunk_size)
                
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return f"Summarization error: {str(e)}"
    
    async def _summarize_text(self, text: str) -> str:
        """Summarize text using custom prompt."""
        try:
            messages = self.summarize_prompt.format_messages(text=text)
            result = await self.client.ainvoke(messages)
            
            summary = result.content.strip() if result.content else "No summary generated"
            logger.debug(f"Generated summary: {len(summary.split())} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Text summarization failed: {e}")
            return f"Error: {str(e)}"
    
    async def _recursive_summarize(self, text: str, max_chunk_size: int = None) -> str:
        """Recursively summarize text by chunking and combining summaries."""
        if max_chunk_size is None:
            max_chunk_size = self.chunk_size

        try:
            # Split into chunks
            docs = self.text_splitter.create_documents([text])
            logger.info(f"Split document into {len(docs)} chunks")
            
            if len(docs) == 1:
                return await self._summarize_text(docs[0].page_content)
 
            tasks = [self._summarize_text(chunk_doc.page_content) for chunk_doc in docs]
            chunk_summaries = await asyncio.gather(*tasks) [2, 1]

            combined_summaries = " ".join(chunk_summaries)
            
            # If combined summaries are still too long, recursively summarize again
            if len(combined_summaries) > max_chunk_size:
                logger.info(f"Combined summaries too long ({len(combined_summaries)} chars), recursing...")
                return await self._recursive_summarize(combined_summaries, max_chunk_size)
            else:
                logger.info("Final summarization of combined summaries")
                return await self._summarize_text(combined_summaries)
            
        except Exception as e:
            logger.error(f"Recursive summarization failed: {e}")
            return f"Error in recursive summarization: {str(e)}"