from safetensors import safe_open

# 确保文件存在且是有效的 safetensors 文件
file_path = "illustration.safetensors"

try:
    # 使用正确的方法打开和读取 safetensors 文件（假设是 PyTorch 模型）
    with safe_open(file_path, framework="pt") as f:
        model_data = f.load()  # 加载模型或其他张量数据
    print("Safetensors 文件加载成功！")
except Exception as e:
    print(f"加载 safetensors 文件时出错: {e}")
