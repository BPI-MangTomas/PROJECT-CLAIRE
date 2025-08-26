import torch
import sys

print("="*50)
print("PyTorch Installation Check")
print("="*50)
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print("="*50)

# Test device creation
try:
    cpu_device = torch.device('cpu')
    print(f"✅ CPU device created: {cpu_device}")
except Exception as e:
    print(f"❌ CPU device error: {e}")

try:
    cuda_device = torch.device('cuda')
    print(f"✅ CUDA device created: {cuda_device}")
except Exception as e:
    print(f"❌ CUDA device error (expected on CPU-only): {e}")

print("="*50)

# Test tensor creation
try:
    cpu_tensor = torch.tensor([1, 2, 3])
    print(f"✅ CPU tensor created: {cpu_tensor}")
except Exception as e:
    print(f"❌ CPU tensor error: {e}")

print("="*50)
print("Recommendation: Use CPU for all operations")
print("="*50)