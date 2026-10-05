import torch

print("PyTorch 版本:      ", torch.__version__)
print("CUDA 是否可用:     ", torch.cuda.is_available())
print("PyTorch 编译的 CUDA:", torch.version.cuda)  # 若是 CPU 版，返回 None
print("cuDNN 版本:        ", torch.backends.cudnn.version())
print("GPU 数量:          ", torch.cuda.device_count())

if torch.cuda.is_available():
    print("当前 GPU:          ", torch.cuda.get_device_name(0))
    print("GPU 计算能力:      ", torch.cuda.get_device_capability(0))
