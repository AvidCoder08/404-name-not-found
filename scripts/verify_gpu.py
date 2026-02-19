import torch
import sys

def verify_gpu():
    print(f"Python version: {sys.version}")
    print(f"Torch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print("Success: GPU is accessible by PyTorch.")
    else:
        print("Warning: GPU is NOT accessible. PyTorch is running on CPU.")

if __name__ == "__main__":
    verify_gpu()
