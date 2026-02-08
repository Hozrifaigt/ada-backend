from pydantic import BaseModel
from typing import List, Optional, Dict

class ClientProfile(BaseModel):
    name: str
    geography: Optional[str] = None
    industry: Optional[str] = None
    employee_count: Optional[int] = None


class PolicyTopicRequest(BaseModel):
    query: str
    client_profile: Optional[ClientProfile] = None

class PolicyTopicResponse(BaseModel):
    sugested_topics: List[str]
    related_documents: List[str] = []


class PolicyDraftRequest(BaseModel):
    topic: str
    client_profile: Optional[ClientProfile] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

class PolicyDraftRequest(BaseModel):
    # to be configured later 
    policy_id: str
    topic: str
    content: List[Dict[str, str ]]
    
