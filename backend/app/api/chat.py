from fastapi import APIRouter, HTTPException, Depends
from typing import Any
import time
import logging
from app.models import ChatRequest, ChatResponse, LanguageDetection, EmotionDetection, RetrievedContext
from app.dependencies import get_language_detector, get_emotion_detector, get_vector_db, get_answer_generator

logger = logging.getLogger(__name__)
router = APIRouter()

# Default safe values
DEFAULT_LANGUAGE = "english"
DEFAULT_EMOTION = "neutral"
CONFIDENCE_THRESHOLD = 0.3

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    language_detector=Depends(get_language_detector),
    emotion_detector=Depends(get_emotion_detector),
    vector_db=Depends(get_vector_db),
    answer_generator=Depends(get_answer_generator)
) -> Any:
    """Process chat with comprehensive error handling"""
    
    try:
        start_time = time.time()
        
        # Prepare input
        full_question = request.question
        has_attachment = False
        
        if hasattr(request, 'extracted_text') and request.extracted_text:
            has_attachment = True
            # Limit extracted text to prevent overload
            truncated_text = request.extracted_text[:1000]
            full_question = f"{request.question}\n\n[Document Content]:\n{truncated_text}"
            logger.info(f"Processing with attachment ({len(request.extracted_text)} chars)")
        
        # 1. Language Detection with fallback
        try:
            language, lang_confidence = language_detector.predict(request.question)
            if lang_confidence < CONFIDENCE_THRESHOLD:
                logger.warning(f"Low language confidence: {lang_confidence}")
                language = DEFAULT_LANGUAGE
                lang_confidence = 0.5
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            language = DEFAULT_LANGUAGE
            lang_confidence = 0.0
            
        language_result = LanguageDetection(language=language, confidence=lang_confidence)
        
        # 2. Emotion Detection with fallback  
        try:
            emotion, emo_confidence = emotion_detector.predict(request.question)
            if emo_confidence < CONFIDENCE_THRESHOLD:
                logger.warning(f"Low emotion confidence: {emo_confidence}")
                emotion = DEFAULT_EMOTION
                emo_confidence = 0.5
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            emotion = DEFAULT_EMOTION
            emo_confidence = 0.0
            
        emotion_result = EmotionDetection(emotion=emotion, confidence=emo_confidence)
        
        # 3. Knowledge Retrieval with fallback
        contexts = []
        retrieved_docs = []
        try:
            search_query = full_question if has_attachment else request.question
            retrieved_docs = vector_db.search(search_query, top_k=4)
            
            contexts = [
                RetrievedContext(
                    content=doc['content'][:500],  # Limit content size
                    title=doc['title'],
                    score=doc['score'],
                    source=doc.get('source')
                )
                for doc in retrieved_docs
            ]
        except Exception as e:
            logger.error(f"Knowledge retrieval failed: {e}")
            # Continue without contexts
            
        # 4. Answer Generation with fallback
        try:
            # Prepare contexts for generator
            answer_contexts = []
            
            # Add extracted text if available
            if has_attachment and hasattr(request, 'extracted_text'):
                answer_contexts.append({
                    'content': request.extracted_text[:500],
                    'title': 'Uploaded Document',
                    'score': 1.0
                })
            
            # Add retrieved contexts
            if retrieved_docs:
                answer_contexts.extend([
                    {'content': doc['content'], 'title': doc['title'], 'score': doc['score']}
                    for doc in retrieved_docs[:3]
                ])
            
            # Call the answer generator
            if hasattr(answer_generator, 'generate_answer'):
                # Using the new generator with error handling
                answer_result = answer_generator.generate_answer(
                    question=request.question,
                    language=language,
                    emotion=emotion,
                    contexts=answer_contexts if answer_contexts else [],
                    extracted_text=request.extracted_text if hasattr(request, 'extracted_text') else None
                )
                
                # Extract answer from result
                if isinstance(answer_result, dict):
                    answer = answer_result.get('answer', '')
                else:
                    answer = str(answer_result)
            else:
                # Fallback if generator doesn't have the method
                answer = _generate_simple_answer(
                    question=request.question,
                    language=language,
                    emotion=emotion,
                    contexts=answer_contexts
                )
                
            # Validate answer
            if not answer or len(answer.strip()) < 10:
                answer = _get_fallback_answer(language)
                
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            answer = _get_fallback_answer(language)
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            answer=answer,
            language=language_result,
            emotion=emotion_result,
            contexts=contexts,
            processing_time=processing_time,
            has_attachment=has_attachment
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint critical error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return minimal safe response
        return ChatResponse(
            answer="I apologize, but I'm experiencing technical difficulties. Please try again or contact customer service at 889-10000.",
            language=LanguageDetection(language=DEFAULT_LANGUAGE, confidence=0.0),
            emotion=EmotionDetection(emotion=DEFAULT_EMOTION, confidence=0.0),
            contexts=[],
            processing_time=0.0,
            has_attachment=False
        )

def _generate_simple_answer(question: str, language: str, emotion: str, contexts: list) -> str:
    """Simple answer generation fallback"""
    if contexts and len(contexts) > 0:
        context_text = contexts[0].get('content', '')[:400]
        
        if language == 'tagalog':
            answer = f"Batay sa aming impormasyon:\n\n{context_text}"
        elif language == 'taglish':
            answer = f"Based sa our information:\n\n{context_text}"
        else:
            answer = f"Based on our information:\n\n{context_text}"
    else:
        answer = _get_fallback_answer(language)
    
    # Add emotion response
    if emotion in ['frustrated', 'urgent', 'worried']:
        if language == 'tagalog':
            answer += "\n\nNauunawaan namin ang inyong sitwasyon."
        else:
            answer += "\n\nWe understand your concern and are here to help."
    
    return answer

def _get_fallback_answer(language: str) -> str:
    """Get fallback answer based on language"""
    if language == 'tagalog':
        return "Salamat sa inyong tanong. Para sa detalyadong tulong, tumawag sa 889-10000."
    elif language == 'taglish':
        return "Thank you sa iyong question. For detailed help, please call 889-10000."
    else:
        return "Thank you for your question. For detailed assistance, please call 889-10000."