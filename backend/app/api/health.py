from fastapi import APIRouter, Depends
from datetime import datetime
from app.models import HealthResponse
from app.dependencies import get_language_detector, get_emotion_detector, get_vector_db, get_answer_generator
import os
import pytesseract
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health", response_model=HealthResponse)
async def health_check(
    language_detector=Depends(get_language_detector),
    emotion_detector=Depends(get_emotion_detector),
    vector_db=Depends(get_vector_db),
    answer_generator=Depends(get_answer_generator)
):
    """Check health status of all services"""
    
    # Check models
    models_loaded = {
        "language_model": False,
        "emotion_model": False,
        "answer_generator": False
    }
    
    try:
        # Check language model
        models_loaded["language_model"] = (
            hasattr(language_detector, 'model') and 
            language_detector.model is not None
        )
    except Exception as e:
        logger.error(f"Error checking language model: {e}")
        models_loaded["language_model"] = False
    
    try:
        # Check emotion model
        models_loaded["emotion_model"] = (
            hasattr(emotion_detector, 'model') and 
            emotion_detector.model is not None
        )
    except Exception as e:
        logger.error(f"Error checking emotion model: {e}")
        models_loaded["emotion_model"] = False
    
    try:
        # Check answer generator (GGUF model)
        models_loaded["answer_generator"] = (
            answer_generator is not None and
            hasattr(answer_generator, 'model') and
            answer_generator.model is not None
        )
        
        # Log model status
        if models_loaded["answer_generator"]:
            model_path = getattr(answer_generator, 'model_path', 'unknown')
            logger.info(f"Answer generator using: {os.path.basename(model_path)}")
    except Exception as e:
        logger.error(f"Error checking answer generator: {e}")
        models_loaded["answer_generator"] = False
    
    # Check vector DB
    vector_db_ready = False
    try:
        vector_db_ready = (
            vector_db is not None and
            hasattr(vector_db, 'index') and
            vector_db.index is not None
        )
    except Exception as e:
        logger.error(f"Error checking vector DB: {e}")
        vector_db_ready = False
    
    # Check OCR availability
    ocr_available = False
    try:
        # Check if Tesseract is installed
        if os.name == 'nt':  # Windows
            tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            ocr_available = os.path.exists(tesseract_path)
        else:
            # Try to get version to check if installed
            pytesseract.get_tesseract_version()
            ocr_available = True
    except Exception as e:
        logger.debug(f"OCR not available: {e}")
        ocr_available = False
    
    # Determine overall status
    all_models_loaded = all(models_loaded.values())
    
    # Status logic:
    # - healthy: All critical components are loaded
    # - degraded: Some components loaded but not all (can still function)
    # - unhealthy: Critical components missing
    if all_models_loaded and vector_db_ready:
        status = "healthy"
    elif (models_loaded["language_model"] and models_loaded["emotion_model"] and vector_db_ready):
        # Can function without GGUF model (retrieval-only mode)
        status = "degraded"
    else:
        status = "unhealthy"
    
    # Add additional info if in degraded state
    if status == "degraded" and not models_loaded["answer_generator"]:
        logger.warning("Answer generator not loaded - using retrieval-only mode")
    
    return HealthResponse(
        status=status,
        models_loaded=models_loaded,
        vector_db_ready=vector_db_ready,
        ocr_available=ocr_available,
        timestamp=datetime.now()
    )