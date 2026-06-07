import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))

# THE REAL TEST — force an actual computation on the GPU
x = torch.randn(1000, 1000, device="cuda")
y = torch.randn(1000, 1000, device="cuda")
z = x @ y
torch.cuda.synchronize()   # wait for the GPU to actually finish
print("GPU matmul result sum:", z.sum().item())
print("SUCCESS — GPU compute works")