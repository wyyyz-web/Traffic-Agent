import os
from ultralytics import YOLO

# 定义你要横向对比的模型变体列表（根据你的截图，使用 yolo26 系列或 yolov8 系列）
# 这里我们先跑 n(极小), s(小), m(中) 三个量级进行对比
model_list = ['yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt']

print("🎯 准备开启多模型基准测试 (Benchmarking)...")

for model_name in model_list:
    print(f"\n{'='*60}")
    print(f"🚀 正在启动流水线：当前训练模型 -> {model_name}")
    print(f"{'='*60}\n")
    
    # 1. 初始化当前模型
    # 如果本地没有对应的 .pt 文件，Ultralytics 会尝试自动下载
    model = YOLO(model_name)
    
    # 2. 提取不带后缀的模型名，作为保存文件夹的标识 (例如：benchmark_yolo26n)
    run_name = f"benchmark_{model_name.split('.')[0]}"
    
    # 3. 启动训练 (参数与你之前成功的满分训练保持一致)
    results = model.train(
        data='plate.yaml',       # 指向你的 CCPD2020 数据集配置
        epochs=50,               # 统一训练 50 轮，控制变量
        imgsz=640,               # 统一输入图片尺寸
        project='runs/detect',   # 结果保存的主目录
        name=run_name,           # 这次的结果会保存在独立的子目录中
        batch=16,                # 批次大小
        device=0                 # 强制使用 GPU
    )
    
    print(f"\n✅ 模型 {model_name} 训练完毕！各项验证指标已保存至 runs/detect/{run_name}\n")

print("🎉 恭喜！所有基准测试模型均已训练结束，请前往 runs/detect/ 目录查看各模型的性能对比数据！")
