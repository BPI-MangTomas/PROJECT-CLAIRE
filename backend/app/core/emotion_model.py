import torch
from typing import Tuple
import logging
from app.config import settings
from app.core.language_model import AdvancedSingleTaskModel
from transformers import DistilBertTokenizer

logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self):
        # Force CPU if CUDA not available
        device_name = settings.DEVICE
        if device_name == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device_name = "cpu"
            
        self.device = torch.device(device_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        self.model = None
        self.emotion_mapping = {
            0: 'confused', 1: 'frustrated', 2: 'grateful',
            3: 'neutral', 4: 'urgent', 5: 'worried'
        }
        self._load_model()
        
    def _load_model(self):
        try:
            # Always load to CPU first to avoid device mismatch
            logger.info(f"Loading emotion model from {settings.EMOTION_MODEL_PATH}")
            checkpoint = torch.load(
                settings.EMOTION_MODEL_PATH, 
                map_location='cpu'  # Always load to CPU first
            )
            
            config = checkpoint.get("config", {})
            logger.info(f"Model config: {config}")
            
            # Auto-detect use_attention_pooling
            state_keys = checkpoint["model_state_dict"].keys()
            use_attention_pooling = any(k.startswith("attention.") for k in state_keys)
            logger.info(f"Detected attention pooling: {use_attention_pooling}")
            
            # Create model on CPU first
            self.model = AdvancedSingleTaskModel(
                num_classes=6,
                dropout_rate=0.3,
                freeze_layers=0,
                use_attention_pooling=use_attention_pooling
            )
            
            # Load state dict (still on CPU)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Only move to device if it's actually available
            if self.device.type == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(self.device)
            else:
                self.model = self.model.to('cpu')
                self.device = torch.device('cpu')
                
            self.model.eval()
            
            logger.info(f"Emotion model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading emotion model: {e}")
            raise
            
    def predict(self, text: str) -> Tuple[str, float]:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=settings.MAX_LENGTH
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            probs = torch.softmax(logits, dim=1)
            idx = probs.argmax(dim=1).item()
            
        return self.emotion_mapping[idx], probs[0][idx].item()