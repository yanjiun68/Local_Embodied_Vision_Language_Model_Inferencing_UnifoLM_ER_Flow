import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

free, total = torch.cuda.mem_get_info()

print(f"Free VRAM:  {free / 1024**3:.2f} GB")
print(f"Total VRAM: {total / 1024**3:.2f} GB")
