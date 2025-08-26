from fastapi import APIRouter, HTTPException, Depends
from typing import Any
import time
import logging
from app.models import ChatRequest, ChatResponse, LanguageDetection, EmotionDetection, RetrievedContext
from app.dependencies import get_language_detector, get_emotion_detector, get_vector_db, get_answer_generator
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

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
            truncated_text = request.extracted_text[:settings.MAX_RESPONSE_LENGTH]
            full_question = f"{request.question}\n\n[Document Content]:\n{truncated_text}"
            logger.info(f"Processing with attachment ({len(request.extracted_text)} chars)")
        
        # 1. Language Detection with fallback
        language = settings.DEFAULT_LANGUAGE
        lang_confidence = 0.0
        
        if settings.ENABLE_LANGUAGE_DETECTION:
            try:
                language, lang_confidence = language_detector.predict(request.question)
                if lang_confidence < settings.LANGUAGE_CONFIDENCE_THRESHOLD:
                    logger.warning(f"Low language confidence: {lang_confidence}")
                    language = settings.DEFAULT_LANGUAGE
                    lang_confidence = 0.5
            except Exception as e:
                logger.error(f"Language detection failed: {e}")
                language = settings.DEFAULT_LANGUAGE
                lang_confidence = 0.0
        
        language_result = LanguageDetection(language=language, confidence=lang_confidence)
        
        # 2. Emotion Detection with fallback  
        emotion = settings.DEFAULT_EMOTION
        emo_confidence = 0.0
        
        if settings.ENABLE_EMOTION_DETECTION:
            try:
                emotion, emo_confidence = emotion_detector.predict(request.question)
                if emo_confidence < settings.EMOTION_CONFIDENCE_THRESHOLD:
                    logger.warning(f"Low emotion confidence: {emo_confidence}")
                    emotion = settings.DEFAULT_EMOTION
                    emo_confidence = 0.5
            except Exception as e:
                logger.error(f"Emotion detection failed: {e}")
                emotion = settings.DEFAULT_EMOTION
                emo_confidence = 0.0
        
        emotion_result = EmotionDetection(emotion=emotion, confidence=emo_confidence)
        
        # 3. Knowledge Retrieval with fallback
        contexts = []
        retrieved_docs = []
        
        if settings.ENABLE_CONTEXT_SEARCH:
            try:
                search_query = full_question if has_attachment else request.question
                retrieved_docs = vector_db.search(search_query, top_k=settings.TOP_K)
                
                # We'll create contexts for the vector search, but may not use them
                # if the answer generator determines this is a greeting
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
                    for doc in retrieved_docs[:settings.TOP_K]
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
                
                # Extract answer and contexts from result
                if isinstance(answer_result, dict):
                    answer = answer_result.get('answer', '')
                    
                    # Check if this was a greeting (no contexts should be shown)
                    if answer_result.get('method') in ['greeting_response', 'no_context']:
                        # Don't show contexts for greetings or when no context found
                        contexts = []
                        logger.info(f"Response method: {answer_result.get('method')} - no contexts shown")
                    else:
                        # Use the contexts that were actually used for generation
                        # Convert back to RetrievedContext objects
                        result_contexts = answer_result.get('contexts', [])
                        if result_contexts:
                            contexts = [
                                RetrievedContext(
                                    content=ctx.get('content', '')[:500],
                                    title=ctx.get('title', 'Unknown'),
                                    score=ctx.get('score', 0.0),
                                    source=ctx.get('source')
                                )
                                for ctx in result_contexts[:settings.TOP_K]
                            ]
                        else:
                            # Fallback to original contexts if not in result
                            pass  # Keep the contexts we already have
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
                answer = _get_fallback_answer(language, settings.CUSTOMER_SERVICE_NUMBER)
                contexts = []  # No contexts for fallback
                
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            answer = _get_fallback_answer(language, settings.CUSTOMER_SERVICE_NUMBER)
            contexts = []  # No contexts on error
        
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
            answer=f"I apologize, but I'm experiencing technical difficulties. Please try again or contact customer service at {settings.CUSTOMER_SERVICE_NUMBER}.",
            language=LanguageDetection(language=settings.DEFAULT_LANGUAGE, confidence=0.0),
            emotion=EmotionDetection(emotion=settings.DEFAULT_EMOTION, confidence=0.0),
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
        answer = _get_fallback_answer(language, settings.CUSTOMER_SERVICE_NUMBER)
    
    # Add emotion response
    if emotion in ['frustrated', 'urgent', 'worried']:
        if language == 'tagalog':
            answer += "\n\nNauunawaan namin ang inyong sitwasyon."
        else:
            answer += "\n\nWe understand your concern and are here to help."
    
    return answer

def _get_fallback_answer(language: str, customer_service_number: str) -> str:
    """Get fallback answer based on language"""
    if language == 'tagalog':
        return f"Salamat sa inyong tanong. Para sa detalyadong tulong, tumawag sa {customer_service_number}."
    elif language == 'taglish':
        return f"Thank you sa iyong question. For detailed help, please call {customer_service_number}."
    else:
        return f"Thank you for your question. For detailed assistance, please call {customer_service_number}."