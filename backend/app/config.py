import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, computed_field
import torch
import multiprocessing

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "CLAIRE-RAG [BACKEND]"
    VERSION: str = "1.0.0"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: list = ["*"]
    
    # Base paths
    BASE_PATH: Path = Path(__file__).parent.parent
    KNOWLEDGE_BASE_PATH: str = Field(default_factory=lambda: str(Path(__file__).parent.parent / "knowledge_base"))
    VECTOR_STORE_PATH: str = Field(default_factory=lambda: str(Path(__file__).parent.parent / "vector_store"))
    
    # Model Paths - these will be loaded from env
    CLAIRE_MODEL_Q4_PATH: str = Field(default_factory=lambda: str(Path(__file__).parent.parent / "models/claire_v1.0.0_q4_k_m.gguf"))
    CLAIRE_MODEL_F16_PATH: str = Field(default_factory=lambda: str(Path(__file__).parent.parent / "models/claire_v1.0.0_f16.gguf"))
    LANGUAGE_MODEL_PATH: str = Field(default_factory=lambda: str(Path(__file__).parent.parent / "models/distilbert_language.pt"))
    EMOTION_MODEL_PATH: str = Field(default_factory=lambda: str(Path(__file__).parent.parent / "models/distilbert_emotion.pt"))
    
    # Auto model selection
    AUTO_SELECT_MODEL: bool = True
    
    # Model Settings
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    MAX_LENGTH: int = 512
    TOP_K: int = 4
    
    # GGUF Model Settings
    MODEL_CONTEXT_SIZE: int = 2048
    MODEL_MAX_TOKENS: int = 1024
    MODEL_TEMPERATURE: float = 0.3
    MODEL_TOP_P: float = 0.9
    MODEL_REPEAT_PENALTY: float = 1.1
    MODEL_N_BATCH: int = 512  # For GPU
    MODEL_N_BATCH_CPU: int = 256  # For CPU
    
    # GPU Settings
    GPU_LAYERS: int = 35
    
    # CPU Settings
    F16_KV_CPU: bool = False  # Use fp32 for KV cache on CPU
    USE_MMAP: bool = True
    USE_MLOCK: bool = False
    
    # Timeout Settings
    REQUEST_TIMEOUT: int = 300
    OCR_TIMEOUT: int = 30
    VECTOR_SEARCH_TIMEOUT: int = 30
    GENERATION_TIMEOUT_COOLDOWN: int = 60
    
    # Response Settings
    MAX_RESPONSE_LENGTH: int = 1000
    SHORT_MESSAGE_THRESHOLD: int = 20
    ENABLE_GREETING_DETECTION: bool = True
    
    # Worker Settings
    MAX_WORKERS: int = 2
    BATCH_SIZE: int = 1
    
    # Cache Settings
    ENABLE_MODEL_CACHE: bool = True
    USE_CACHE: bool = True
    
    # Skip model loading for testing
    SKIP_MODEL_LOADING: bool = False
    
    # File upload settings
    MAX_FILE_SIZE: int = 5242880  # 5MB
    
    # Device Detection - computed properties
    @property
    def DEVICE(self) -> str:
        """Auto-detect device with override support"""
        use_cuda = os.environ.get("USE_CUDA", "auto").lower()
        
        # Auto-detect
        if use_cuda == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        
        # Manual override
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
    def MODEL_INFERENCE_TIMEOUT(self) -> int:
        """Timeout based on device"""
        if self.DEVICE == "cuda":
            timeout = os.environ.get("MODEL_INFERENCE_TIMEOUT", "120")
        else:
            timeout = os.environ.get("MODEL_INFERENCE_TIMEOUT_CPU", "300")
        
        try:
            return int(timeout)
        except:
            return 300 if self.DEVICE == "cpu" else 120
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        # Allow extra fields from env that we handle as properties
        extra = "ignore"

settings = Settings()