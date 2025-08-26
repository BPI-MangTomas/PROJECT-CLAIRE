from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class ChatRequest(BaseModel):
    question: str = Field(..., description="User's question")
    session_id: Optional[str] = Field(None, description="Session identifier")
    extracted_text: Optional[str] = Field(None, description="Text extracted from uploaded file")
    
class FileUploadResponse(BaseModel):
    extracted_text: str
    filename: str
    file_type: str
    char_count: int
    processing_time: float
    
class LanguageDetection(BaseModel):
    language: str
    confidence: float
    
class EmotionDetection(BaseModel):
    emotion: str
    confidence: float
    
class RetrievedContext(BaseModel):
    content: str
    title: str
    score: float
    source: Optional[str] = None
    
class ChatResponse(BaseModel):
    answer: str
    language: LanguageDetection
    emotion: EmotionDetection
    contexts: List[RetrievedContext]
    processing_time: float
    has_attachment: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)
    
class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    vector_db_ready: bool
    ocr_available: bool = False  # Make it optional with default value
    timestamp: datetime