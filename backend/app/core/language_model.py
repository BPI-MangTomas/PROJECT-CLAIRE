import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel
from typing import Tuple
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class AdvancedSingleTaskModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3, freeze_layers=0,
                 use_attention_pooling=True):
        super(AdvancedSingleTaskModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
        hidden_size = self.distilbert.config.hidden_size
        self.use_attention_pooling = use_attention_pooling
        
        if freeze_layers > 0:
            for param in self.distilbert.embeddings.parameters():
                param.requires_grad = False
            for layer_idx in range(min(freeze_layers, 6)):
                for param in self.distilbert.transformer.layer[layer_idx].parameters():
                    param.requires_grad = False
                    
        if use_attention_pooling:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1, bias=False)
            )
            
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(hidden_size // 4, num_classes)
        )
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def attention_pooling(self, hidden_states, attention_mask):
        attention_scores = self.attention(hidden_states).squeeze(-1)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)
        return (hidden_states * attention_weights).sum(dim=1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        if self.use_attention_pooling:
            pooled_output = self.attention_pooling(hidden_states, attention_mask)
        else:
            pooled_output = hidden_states[:, 0]
            
        pre_output = F.relu(self.pre_classifier(pooled_output))
        combined = pre_output + pooled_output
        return self.classifier(combined)

class LanguageDetector:
    def __init__(self):
        # Force CPU if CUDA not available
        device_name = settings.DEVICE
        if device_name == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device_name = "cpu"
            
        self.device = torch.device(device_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
        self.model = None
        self.language_mapping = {0: 'english', 1: 'tagalog', 2: 'taglish'}
        self._load_model()
        
    def _load_model(self):
        try:
            # Always load to CPU first to avoid device mismatch
            logger.info(f"Loading language model from {settings.LANGUAGE_MODEL_PATH}")
            checkpoint = torch.load(
                settings.LANGUAGE_MODEL_PATH, 
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
                num_classes=3,
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
            
            logger.info(f"Language model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading language model: {e}")
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
            
        return self.language_mapping[idx], probs[0][idx].item()