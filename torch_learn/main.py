import torch

# 查看 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 检查 CUDA 是否可用
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

# 如果有 GPU，查看 GPU 信息
if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")