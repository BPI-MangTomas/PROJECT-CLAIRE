from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
import asyncio
from app.config import settings
from app.api import chat, health, upload
from app.core.knowledge_base import KnowledgeBaseProcessor
from app.dependencies import get_vector_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up CLAIRE-RAG [BACKEND]...")
    
    # Set longer timeout for asyncio tasks
    asyncio.get_event_loop().set_debug(False)
    
    # Initialize knowledge base and vector database
    try:
        kb_processor = KnowledgeBaseProcessor(settings.KNOWLEDGE_BASE_PATH)
        documents = kb_processor.process_all_files()
        
        vector_db = get_vector_db()
        vector_db.build_index(documents)
        
        logger.info("Knowledge base and vector database initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down CLAIRE RAG Backend...")

# Create FastAPI app with custom settings
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
    # Extended timeout for docs
    swagger_ui_parameters={"docExpansion": "none", "defaultModelsExpandDepth": -1}
)

# Add timeout middleware
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        # Set 5 minute timeout for all requests
        response = await asyncio.wait_for(call_next(request), timeout=300.0)
        return response
    except asyncio.TimeoutError:
        logger.error(f"Request timeout for {request.url}")
        return JSONResponse(
            status_code=504,
            content={"detail": "Request processing timeout. This is normal for CPU processing - please try again."}
        )

# Set up CORS with longer max_age
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600  # Cache CORS for 1 hour
)

# Include routers
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["chat"])
app.include_router(upload.router, prefix=f"{settings.API_V1_STR}/upload", tags=["upload"])
app.include_router(health.router, prefix=f"{settings.API_V1_STR}", tags=["health"])

@app.get("/")
async def root():
    return {
        "message": "CLAIRE-RAG API [BACKEND]",
        "version": settings.VERSION,
        "mode": "CPU Processing - Extended Timeouts",
        "docs": "/docs"
    }