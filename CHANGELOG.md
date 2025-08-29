# Changelog

All notable changes to the CLAIRE-RAG [BACKEND] project will be documented in this file.

## [1.1.0] - 2025-08-30

### 🚀 Added
- Comprehensive environment variable configuration system - all configuration values now sourced from `.env`
- JSON array parsing support for list-type environment variables (CORS origins, file formats, etc.)
- New configuration options:
  - OCR settings (`TESSERACT_PATH`, `OCR_MAX_IMAGE_SIZE`, `OCR_MIN_CONFIDENCE`, `OCR_LANGUAGE`)
  - Advanced model penalties (`FREQUENCY_PENALTY`, `PRESENCE_PENALTY`)
  - Model loading settings (`MODEL_SEED`, `LOGITS_ALL`, `VOCAB_ONLY`, `USE_EMBEDDING`, `MODEL_VERBOSE`)
  - Response formatting (`MAX_SENTENCES_PER_RESPONSE`, `PARAGRAPH_BREAK_LENGTH`, `SENTENCE_SIMILARITY_THRESHOLD`)
  - Server settings (`HOST`, `PORT`, `RELOAD`, `TIMEOUT_KEEP_ALIVE`, `TIMEOUT_GRACEFUL_SHUTDOWN`)
  - Health check settings (`HEALTH_CHECK_INTERVAL`, `DEGRADED_MODE_ENABLED`)
  - Security settings (`ALLOWED_HOSTS`, `TRUSTED_HOSTS`, `DISABLE_DOCS`, `DISABLE_REDOC`)
  - Rate limiting options (`RATE_LIMIT_ENABLED`, `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_PERIOD`)
  - Development/debug flags (`DEBUG_MODE`, `SHOW_PROMPTS`, `SHOW_CONTEXTS`, `MEASURE_PERFORMANCE`)
- Feature flags for toggling functionality:
  - `ENABLE_OCR`
  - `ENABLE_FILE_UPLOAD`
  - `ENABLE_CONTEXT_SEARCH`
  - `ENABLE_EMOTION_DETECTION`
  - `ENABLE_LANGUAGE_DETECTION`
- Dynamic property-based configuration for device detection and model selection
- Proper Pydantic v2 field validators for environment variable parsing

### 🔄 Changed

#### Model Configuration Updates
- **Context window increased**: `MODEL_CONTEXT_SIZE` from 2048 → 8192 tokens
  - Allows processing longer documents and maintaining more conversation context
- **Temperature adjusted**: `MODEL_TEMPERATURE` from 0.3 → 0.6
  - More creative and varied responses while maintaining coherence
- **Repeat penalty increased**: `MODEL_REPEAT_PENALTY` from 1.1 → 1.3
  - Stronger prevention of repetitive text patterns
- **Model max tokens**: Remains at 1024 (optimal for response length)
- **Top-p sampling**: Remains at 0.9 (good balance of quality and diversity)

#### Refactoring
- **config.py**: Complete overhaul with proper environment variable loading
- **ocr_processor.py**: Removed all hardcoded values, now uses settings
- **upload.py**: Dynamic configuration for file size limits and timeouts
- **chat.py**: Configurable confidence thresholds and default values
- **logger.py**: Environment-based logging configuration
- **answer_generator.py**: Fallback settings now read from environment variables

### 🐛 Fixed
- Fixed hardcoded Tesseract OCR path that caused Windows-specific issues
- Resolved fallback Settings class in `answer_generator.py` not reading from environment
- Fixed inconsistency between `SHORT_MESSAGE_THRESHOLD` and `GREETING_MAX_LENGTH`
- Corrected hardcoded values in text processing functions:
  - Sentence limit in `_ensure_conciseness()`
  - Paragraph break length in `_format_paragraphs()`
  - Similarity threshold in `_is_similar()`
- Fixed JSON array parsing for list-type environment variables
- Resolved device detection issues when CUDA requested but unavailable

### 🔧 Technical Improvements
- Better error handling with environment variable fallbacks
- Improved modularity with centralized configuration management
- Enhanced maintainability through elimination of hardcoded values
- More robust timeout handling for model inference
- Proper thread configuration for CPU optimization
- Dynamic batch size selection based on device (GPU vs CPU)

### 📝 Configuration
- All configuration now centralized in `.env` file
- Support for both JSON and comma-separated list formats in environment variables
- Auto-detection of hardware capabilities with manual override options
- Comprehensive logging configuration through environment variables

### 🏗️ Infrastructure
- Version bumped to 1.1.0
- Added proper Config class with Pydantic v2 compatibility
- Implemented computed properties for dynamic configuration values
- Enhanced settings validation with proper type checking

## [1.0.0] - 2025-08-25

### Initial Release
- CLAIRE (Conversational Language AI for Resolution & Engagement) RAG system
- Multi-language support (English, Tagalog, Taglish)
- Emotion detection and context-aware responses
- GGUF model support with GPU acceleration
- OCR capabilities for document processing
- Vector database for knowledge retrieval
- FastAPI-based REST API
- Basic configuration with `.env` file support

### Model Configuration (v1.0.0)
- `MODEL_CONTEXT_SIZE`: 2048 tokens
- `MODEL_MAX_TOKENS`: 1024 tokens
- `MODEL_TEMPERATURE`: 0.3
- `MODEL_TOP_P`: 0.9
- `MODEL_REPEAT_PENALTY`: 1.1

---

## Migration Guide (v1.0.0 → v1.1.0)

### Environment Variable Changes

1. **Model Configuration Updates**
   ```env
   # Old (v1.0.0)
   MODEL_CONTEXT_SIZE=2048
   MODEL_TEMPERATURE=0.3
   MODEL_REPEAT_PENALTY=1.1
   
   # New (v1.1.0)
   MODEL_CONTEXT_SIZE=8192
   MODEL_TEMPERATURE=0.6
   MODEL_REPEAT_PENALTY=1.3
   ```

2. **New Required Variables**
   - Add OCR configuration if using OCR features
   - Configure server settings for production deployment
   - Set appropriate feature flags based on your use case

3. **List Format Variables**
   - Can now use JSON format: `["item1", "item2"]`
   - Or comma-separated: `item1,item2,item3`

### Code Changes

No breaking API changes. Internal refactoring should be transparent to API consumers.

### Performance Notes

- Increased context window may require more RAM (approximately 2-4GB additional)
- Higher temperature setting produces more varied responses
- Stronger repeat penalty improves response quality but may slightly increase generation time

---

## Support

Got questions or ran into an issue? Drop it here:
- Email Address: claire.bot2025@gmail.com
- GitHub Issues: BPI-MangTomas/PROJECT-CLAIRE/issues
