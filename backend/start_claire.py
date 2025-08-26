"""
Quick start script for CLAIRE with error checking and auto GPU/CPU detection
"""
import os
import sys
import subprocess
import time
import multiprocessing
import torch
from pathlib import Path
from dotenv import load_dotenv

# Ensure backend folder is in Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)  # Make app/ importable

# Load environment variables
env_path = Path(script_dir) / '.env'
load_dotenv(env_path)

def detect_hardware():
    """Detect available hardware (GPU/CPU) and set environment accordingly"""
    cpu_count = multiprocessing.cpu_count()
    has_gpu = torch.cuda.is_available()
    
    if has_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"✓ GPU detected: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
        
        # Check if F16 model exists
        f16_model = Path(script_dir) / "models" / "claire_v1.0.0_f16.gguf"
        if f16_model.exists():
            print(f"  Using F16 model for GPU: {f16_model.name}")
        else:
            print(f"  F16 model not found, will use Q4 quantized model")
    else:
        print(f"✓ CPU mode: {cpu_count} cores available")
        print(f"  Using Q4 quantized model for efficient CPU inference")
    
    return has_gpu, cpu_count

def optimize_settings():
    """Set environment variables for optimal performance based on hardware"""
    has_gpu, cpu_count = detect_hardware()
    
    # Read current env settings
    use_cuda = os.environ.get("USE_CUDA", "auto").lower()
    
    # Auto-detection mode
    if use_cuda == "auto":
        if has_gpu:
            os.environ["USE_CUDA"] = "true"
            os.environ["DEVICE"] = "cuda"
            print(f"✓ Auto-configured for GPU acceleration")
        else:
            os.environ["USE_CUDA"] = "false"
            os.environ["DEVICE"] = "cpu"
            print(f"✓ Auto-configured for CPU optimization")
    
    # Thread optimization for CPU operations
    threads = os.environ.get("OMP_NUM_THREADS", "auto")
    if threads == "auto":
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(cpu_count)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
        
    # Set llama-cpp threads if auto
    llama_threads = os.environ.get("LLAMA_CPP_THREADS", "auto")
    if llama_threads == "auto":
        os.environ["LLAMA_CPP_THREADS"] = str(max(cpu_count - 1, 1))
    
    # Disable GPU if not available but requested
    if use_cuda == "true" and not has_gpu:
        print("⚠ GPU requested but not available, falling back to CPU")
        os.environ["USE_CUDA"] = "false"
        os.environ["DEVICE"] = "cpu"
    
    # Additional optimizations
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
    os.environ["SKIP_MODEL_LOADING"] = "false"  # Ensure model loads
    
    # Timeout adjustments based on device
    if os.environ.get("DEVICE") == "cuda":
        if not os.environ.get("MODEL_INFERENCE_TIMEOUT"):
            os.environ["MODEL_INFERENCE_TIMEOUT"] = "120"  # 2 minutes for GPU
    else:
        if not os.environ.get("MODEL_INFERENCE_TIMEOUT_CPU"):
            os.environ["MODEL_INFERENCE_TIMEOUT_CPU"] = "300"  # 5 minutes for CPU
    
    print(f"✓ Environment optimized for {os.environ.get('DEVICE', 'cpu').upper()} processing")

def check_requirements():
    """Check if all requirements are met"""
    issues = []
    warnings = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
    
    # Check critical packages
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ PyTorch installed with CUDA support")
        else:
            print(f"✓ PyTorch installed (CPU mode)")
    except ImportError:
        issues.append("PyTorch not installed")
    
    try:
        import fastapi
        print("✓ FastAPI installed")
    except ImportError:
        issues.append("FastAPI not installed")
    
    # Check llama-cpp-python
    try:
        from llama_cpp import Llama
        print("✓ llama-cpp-python installed")
    except ImportError:
        print("✗ llama-cpp-python not installed")
        print("  Install with: pip install llama-cpp-python==0.2.77")
        issues.append("llama-cpp-python not installed")
    
    # Check Tesseract (for OCR)
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        print("✓ Tesseract OCR found")
    else:
        print("⚠ Tesseract OCR not found - OCR features for images will not work")
        print("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        warnings.append("Tesseract unavailable")
    
    # Check models
    models_dir = Path(script_dir) / "models"
    if os.path.exists(models_dir):
        # Check for GGUF models
        q4_model = models_dir / "claire_v1.0.0_q4_k_m.gguf"
        f16_model = models_dir / "claire_v1.0.0_f16.gguf"
        
        models_found = []
        if q4_model.exists():
            size_gb = q4_model.stat().st_size / (1024**3)
            models_found.append(f"Q4 model ({size_gb:.1f} GB)")
        
        if f16_model.exists():
            size_gb = f16_model.stat().st_size / (1024**3)
            models_found.append(f"F16 model ({size_gb:.1f} GB)")
        
        if models_found:
            print(f"✓ GGUF models found: {', '.join(models_found)}")
        else:
            print("✗ No GGUF model files found in models/")
            issues.append("No GGUF model files found")
            
        # Check for DistilBERT models
        lang_model = models_dir / "distilbert_language.pt"
        emo_model = models_dir / "distilbert_emotion.pt"
        
        if lang_model.exists() and emo_model.exists():
            print("✓ Language and Emotion models found")
        else:
            missing = []
            if not lang_model.exists():
                missing.append("distilbert_language.pt")
            if not emo_model.exists():
                missing.append("distilbert_emotion.pt")
            print(f"✗ Missing models: {', '.join(missing)}")
            issues.extend([f"{m} not found" for m in missing])
    else:
        issues.append(f"Models directory not found: {models_dir}")
    
    return issues, warnings

def display_configuration():
    """Display current configuration from environment"""
    print("\n" + "="*50)
    print("CONFIGURATION SUMMARY")
    print("="*50)
    
    device = os.environ.get("DEVICE", "cpu")
    use_cuda = os.environ.get("USE_CUDA", "false")
    
    print(f"Device Mode: {device.upper()}")
    print(f"CUDA Enabled: {use_cuda}")
    
    if device == "cuda":
        gpu_layers = os.environ.get("GPU_LAYERS", "35")
        print(f"GPU Layers to Offload: {gpu_layers}")
        model_path = os.environ.get("CLAIRE_MODEL_F16_PATH", "models/claire_v1.0.0_f16.gguf")
        print(f"Model: F16 (GPU optimized)")
    else:
        threads = os.environ.get("LLAMA_CPP_THREADS", "auto")
        if threads == "auto":
            threads = str(max(multiprocessing.cpu_count() - 1, 1))
        print(f"CPU Threads: {threads}")
        model_path = os.environ.get("CLAIRE_MODEL_Q4_PATH", "models/claire_v1.0.0_q4_k_m.gguf")
        print(f"Model: Q4_K_M (CPU optimized)")
    
    print(f"Model Path: {model_path}")
    
    # Display features
    print("\nFeatures:")
    greeting_detection = os.environ.get("ENABLE_GREETING_DETECTION", "true")
    print(f"  Greeting Detection: {greeting_detection}")
    cache = os.environ.get("ENABLE_MODEL_CACHE", "true")
    print(f"  Model Caching: {cache}")
    
    print("="*50)

def start_server():
    """Start the FastAPI server with optimized settings"""
    print("\n" + "="*50)
    print("Starting CLAIRE [BACKEND] - Auto-Optimized Mode")
    print("="*50)
    
    # Optimize settings based on hardware
    optimize_settings()
    
    # Display configuration
    display_configuration()
    
    try:
        # Determine workers based on device
        workers = "1"  # Single worker for model consistency
        
        print(f"\nStarting server on http://0.0.0.0:8000")
        print("API Docs available at: http://localhost:8000/docs")
        
        if os.environ.get("DEVICE") == "cuda":
            print("\n📊 GPU Mode: Responses should be faster")
            print("   First generation may take 30-60 seconds to warm up")
        else:
            print("\n📊 CPU Mode: Be patient with first generation")
            print("   First generation may take 2-3 minutes to warm up")
            print("   Subsequent generations will be faster")
        
        print("\n💡 Tip: Simple greetings like 'Hi' or 'Hello' get instant responses!")
        print("Press Ctrl+C to stop the server\n")
        
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info",
            "--timeout-keep-alive", "300",
            "--timeout-graceful-shutdown", "30",
            "--workers", workers,
            "--limit-concurrency", "10"
        ], cwd=script_dir)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    print("CLAIRE [BACKEND] Startup Check")
    print("="*50)
    
    # Check environment file
    env_path = Path(script_dir) / '.env'
    if not env_path.exists():
        print("⚠ .env file not found, using default settings")
    else:
        print("✓ .env file loaded")
    
    issues, warnings = check_requirements()
    
    if issues:
        print("\n⚠ Critical issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before starting.")
        sys.exit(1)
    else:
        print("\n✓ All critical checks passed!")
        
        if warnings:
            print("\n⚠ Warnings (non-critical):")
            for warning in warnings:
                print(f"  - {warning}")
        
        time.sleep(2)
        start_server()