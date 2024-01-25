import torch

# 查看 PyTorch 版本
print("PyTorch 版本:", torch.__version__)


# 检查是否支持 CUDA
if torch.cuda.is_available():
    # 创建一个张量并将其移动到 GPU 上
    device = torch.device("cuda")
    x = torch.rand(3, 3).to(device)
    print("CUDA 工作正常")
else:
    print("CUDA 不可用")
