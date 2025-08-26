from functools import lru_cache
from app.core.language_model import LanguageDetector
from app.core.emotion_model import EmotionDetector
from app.core.vector_database import VectorDatabase
from app.core.answer_generator import AnswerGenerator
from app.config import settings

@lru_cache()
def get_language_detector():
    return LanguageDetector()

@lru_cache()
def get_emotion_detector():
    return EmotionDetector()

@lru_cache()
def get_vector_db():
    return VectorDatabase()

@lru_cache()
def get_answer_generator():
    return AnswerGenerator()