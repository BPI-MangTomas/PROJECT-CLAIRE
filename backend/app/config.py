import os
import json
from pathlib import Path
from typing import Optional, List, Any
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ValidationInfo
import torch
import multiprocessing

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "CLAIRE-RAG [BACKEND]"
    VERSION: str = "1.1.0"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: list = ["*"]
    
    # Base paths
    BASE_PATH: Path = Path(__file__).parent.parent
    KNOWLEDGE_BASE_PATH: str = Field(default_factory=lambda: str(Path(__file__).parent.parent / "knowledge_base"))
    VECTOR_STORE_PATH: str = Field(default_factory=lambda: str(Path(__file__).parent.parent / "vector_store"))
    LOGS_PATH: str = Field(default_factory=lambda: str(Path(__file__).parent.parent / "logs"))
    MODELS_PATH: str = Field(default_factory=lambda: str(Path(__file__).parent.parent / "models"))
    
    # Model Paths
    CLAIRE_MODEL_Q4_PATH: str = Field(default_factory=lambda: str(Path(__file__).parent.parent / "models/claire_v1.0.0_q4_k_m.gguf"))
    CLAIRE_MODEL_F16_PATH: str = Field(default_factory=lambda: str(Path(__file__).parent.parent / "models/claire_v1.0.0_f16.gguf"))
    LANGUAGE_MODEL_PATH: str = Field(default_factory=lambda: str(Path(__file__).parent.parent / "models/distilbert_language.pt"))
    EMOTION_MODEL_PATH: str = Field(default_factory=lambda: str(Path(__file__).parent.parent / "models/distilbert_emotion.pt"))
    
    # Model Selection
    AUTO_SELECT_MODEL: bool = True
    SKIP_MODEL_LOADING: bool = False
    
    # Model Settings
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    MAX_LENGTH: int = 512
    TOP_K: int = 4
    
    # GGUF Model Settings
    MODEL_CONTEXT_SIZE: int = 8192
    MODEL_MAX_TOKENS: int = 1024
    MODEL_TEMPERATURE: float = 0.6
    MODEL_TOP_P: float = 0.9
    MODEL_REPEAT_PENALTY: float = 1.3
    MODEL_N_BATCH: int = 512  # For GPU
    MODEL_N_BATCH_CPU: int = 256  # For CPU
    
    # Additional generation penalties
    FREQUENCY_PENALTY: float = 0.1
    PRESENCE_PENALTY: float = 0.1
    
    # Model loading settings
    MODEL_SEED: int = -1
    LOGITS_ALL: bool = False
    VOCAB_ONLY: bool = False
    USE_EMBEDDING: bool = False
    MODEL_VERBOSE: bool = False
    
    # GPU Settings
    GPU_LAYERS: int = 35
    N_GPU_THREADS: int = 8
    
    # CPU Settings
    F16_KV_CPU: bool = False
    USE_MMAP: bool = True
    USE_MLOCK: bool = False
    
    # Tokenizer settings
    TOKENIZERS_PARALLELISM: bool = False
    
    # Timeout Settings
    REQUEST_TIMEOUT: int = 300
    OCR_TIMEOUT: int = 30
    VECTOR_SEARCH_TIMEOUT: int = 30
    MODEL_INFERENCE_TIMEOUT: int = 120  # For GPU
    MODEL_INFERENCE_TIMEOUT_CPU: int = 300  # For CPU
    GENERATION_TIMEOUT_COOLDOWN: int = 60
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 5242880  # 5MB
    SUPPORTED_IMAGE_FORMATS: List[str] = Field(default_factory=lambda: [".png",".jpg",".jpeg",".gif",".bmp",".tiff"])
    SUPPORTED_DOC_FORMATS: List[str] = Field(default_factory=lambda: [".pdf",".docx",".txt",".xlsx"])
    
    # OCR Settings
    OCR_MAX_IMAGE_SIZE: int = 1500
    OCR_MIN_CONFIDENCE: int = 30
    OCR_LANGUAGE: str = "eng"
    TESSERACT_PATH: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    # Response Settings
    MAX_RESPONSE_LENGTH: int = 1000
    SHORT_MESSAGE_THRESHOLD: int = 20
    MAX_SENTENCES_PER_RESPONSE: int = 5
    PARAGRAPH_BREAK_LENGTH: int = 150
    SENTENCE_SIMILARITY_THRESHOLD: float = 0.7
    
    # Language Detection
    LANGUAGE_CONFIDENCE_THRESHOLD: float = 0.3
    DEFAULT_LANGUAGE: str = "english"
    
    # Emotion Classification
    EMOTION_CONFIDENCE_THRESHOLD: float = 0.3
    DEFAULT_EMOTION: str = "neutral"
    
    # Greeting Detection
    ENABLE_GREETING_DETECTION: bool = True
    GREETING_MAX_LENGTH: int = 20
    
    # Cache Settings
    ENABLE_MODEL_CACHE: bool = True
    USE_CACHE: bool = True
    CACHE_TTL: int = 3600
    
    # Worker & Concurrency
    MAX_WORKERS: int = 2
    BATCH_SIZE: int = 1
    UVICORN_WORKERS: int = 1
    LIMIT_CONCURRENCY: int = 10
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_TO_FILE: bool = True
    LOG_TO_CONSOLE: bool = True
    LOG_FILE_MAX_BYTES: int = 10485760  # 10MB
    LOG_FILE_BACKUP_COUNT: int = 5
    USE_JSON_LOGS: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    TIMEOUT_KEEP_ALIVE: int = 300
    TIMEOUT_GRACEFUL_SHUTDOWN: int = 30
    
    # Health Check
    HEALTH_CHECK_INTERVAL: int = 30
    DEGRADED_MODE_ENABLED: bool = True
    
    # Development/Debug
    DEBUG_MODE: bool = False
    SHOW_PROMPTS: bool = False
    SHOW_CONTEXTS: bool = False
    MEASURE_PERFORMANCE: bool = True
    
    # Feature Flags
    ENABLE_OCR: bool = True
    ENABLE_FILE_UPLOAD: bool = True
    ENABLE_CONTEXT_SEARCH: bool = True
    ENABLE_EMOTION_DETECTION: bool = True
    ENABLE_LANGUAGE_DETECTION: bool = True
    
    # Contact Information
    CUSTOMER_SERVICE_NUMBER: str = "(+632) 889-10000"
    COMPANY_NAME: str = "BPI"
    COMPANY_FULL_NAME: str = "Bank of the Philippine Islands"
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 3600
    
    # Security
    ALLOWED_HOSTS: List[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1", "0.0.0.0"])
    TRUSTED_HOSTS: List[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"])
    DISABLE_DOCS: bool = False
    DISABLE_REDOC: bool = False
    
    # Field validators for JSON parsing from environment variables
    @field_validator('BACKEND_CORS_ORIGINS', 'SUPPORTED_IMAGE_FORMATS', 'SUPPORTED_DOC_FORMATS', 'ALLOWED_HOSTS', 'TRUSTED_HOSTS', mode='before')
    @classmethod
    def parse_json_lists(cls, v: Any, info: ValidationInfo) -> List[str]:
        """Parse JSON arrays from environment variables"""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                # Fallback to comma-separated values
                # Remove brackets if present and split
                v = v.strip('[]')
                return [x.strip().strip('"').strip("'") for x in v.split(',') if x.strip()]
        return v
    
    # Device Detection - computed properties
    @property
    def DEVICE(self) -> str:
        """Auto-detect device with override support"""
        use_cuda = os.environ.get("USE_CUDA", "auto").lower()
        device = os.environ.get("DEVICE", "auto").lower()
        
        # If device is explicitly set and not auto, use it
        if device != "auto":
            if device == "cuda" and not torch.cuda.is_available():
                print("WARNING: CUDA requested but not available, falling back to CPU")
                return "cpu"
            return device
        
        # Auto-detect based on USE_CUDA
        if use_cuda == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        elif use_cuda == "true":
            if torch.cuda.is_available():
                return "cuda"
            else:
                print("WARNING: CUDA requested but not available, falling back to CPU")
                return "cpu"
        else:
            return "cpu"
    
    @property
    def OMP_NUM_THREADS(self) -> int:
        """Get OMP threads from environment"""
        env_val = os.environ.get("OMP_NUM_THREADS", "auto")
        if env_val == "auto":
            return multiprocessing.cpu_count()
        try:
            return int(env_val)
        except:
            return multiprocessing.cpu_count()
    
    @property
    def MKL_NUM_THREADS(self) -> int:
        """Get MKL threads from environment"""
        env_val = os.environ.get("MKL_NUM_THREADS", "auto")
        if env_val == "auto":
            return multiprocessing.cpu_count()
        try:
            return int(env_val)
        except:
            return multiprocessing.cpu_count()
    
    @property
    def LLAMA_CPP_THREADS(self) -> Optional[int]:
        """Get LLAMA CPP threads from environment"""
        env_val = os.environ.get("LLAMA_CPP_THREADS", "auto")
        if env_val == "auto":
            return max(multiprocessing.cpu_count() - 1, 1)
        try:
            return int(env_val) if env_val else None
        except:
            return max(multiprocessing.cpu_count() - 1, 1)
    
    @property
    def CLAIRE_MODEL_PATH(self) -> str:
        """Select appropriate model based on device"""
        if not self.AUTO_SELECT_MODEL:
            # Use Q4 model by default if auto-selection is disabled
            return self.CLAIRE_MODEL_Q4_PATH
        
        # Check if GPU is available and F16 model exists
        if self.DEVICE == "cuda":
            f16_path = Path(self.CLAIRE_MODEL_F16_PATH)
            if f16_path.exists():
                print(f"GPU detected - using F16 model: {f16_path.name}")
                return str(f16_path)
            else:
                print(f"GPU detected but F16 model not found, using Q4 model")
                return self.CLAIRE_MODEL_Q4_PATH
        else:
            print(f"CPU mode - using Q4 quantized model for efficiency")
            return self.CLAIRE_MODEL_Q4_PATH
    
    @property
    def USE_GPU_LAYERS(self) -> int:
        """Number of layers to offload to GPU"""
        if self.DEVICE == "cuda":
            return self.GPU_LAYERS
        return 0
    
    @property
    def MODEL_BATCH_SIZE(self) -> int:
        """Batch size based on device"""
        if self.DEVICE == "cuda":
            return self.MODEL_N_BATCH
        return self.MODEL_N_BATCH_CPU
    
    @property
    def MODEL_INFERENCE_TIMEOUT_DYNAMIC(self) -> int:
        """Timeout based on device"""
        if self.DEVICE == "cuda":
            return self.MODEL_INFERENCE_TIMEOUT
        else:
            return self.MODEL_INFERENCE_TIMEOUT_CPU
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        # Allow extra fields from env that we handle as properties
        extra = "ignore"
        
        @classmethod
        def json_config_sources(cls):
            """Configure JSON parsing for list fields"""
            return []

settings = Settings()