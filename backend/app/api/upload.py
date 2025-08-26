from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Any
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from app.models import FileUploadResponse
from app.core.ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize OCR processor
ocr_processor = OCRProcessor()

# Settings
MAX_FILE_SIZE = 5 * 1024 * 1024  # Reduced to 5MB for faster processing
OCR_TIMEOUT = 30  # seconds

# Thread pool for OCR processing
executor = ThreadPoolExecutor(max_workers=2)

@router.post("/extract-text", response_model=FileUploadResponse)
async def extract_text_from_file(
    file: UploadFile = File(...)
) -> Any:
    """Extract text with timeout and error handling"""
    
    try:
        start_time = time.time()
        
        # Read file
        contents = await file.read()
        
        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE//1024//1024}MB for faster processing"
            )
        
        # Process with timeout
        try:
            # Run OCR in thread pool with timeout
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                executor,
                ocr_processor.process_file,
                contents,
                file.filename
            )
            
            # Wait with timeout
            result = await asyncio.wait_for(future, timeout=OCR_TIMEOUT)
            
        except asyncio.TimeoutError:
            logger.error(f"OCR timeout for {file.filename}")
            raise HTTPException(
                status_code=408,
                detail="Text extraction is taking too long. Please try with a smaller or clearer file."
            )
        
        # Check result
        if not result['success']:
            if result['fallback_used']:
                # Return with warning
                return FileUploadResponse(
                    extracted_text=result['text'],
                    filename=file.filename,
                    file_type=file.filename.split('.')[-1],
                    char_count=len(result['text']),
                    processing_time=result['processing_time'],
                    warning="Text extraction was partially successful"
                )
            else:
                raise HTTPException(
                    status_code=422,
                    detail=f"Failed to extract text: {result['error']}"
                )
        
        processing_time = time.time() - start_time
        
        return FileUploadResponse(
            extracted_text=result['text'],
            filename=file.filename,
            file_type=file.filename.split('.')[-1],
            char_count=len(result['text']),
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process file. Please try again with a different file."
        )