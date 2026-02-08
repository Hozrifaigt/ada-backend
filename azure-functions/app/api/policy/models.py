from typing import Optional
from pydantic import BaseModel

class PolicyGenerationRequest(BaseModel):
    user_query: str
    policy_id: str
    topic_id: str
    subtopic_id: Optional[str] = None

class PolicyGenerationResponse(BaseModel):
    user_query: str
    policy_id: str
    topic_id: str
    subtopic_id: Optional[str]
    topic_content: str
    generated_response: str
