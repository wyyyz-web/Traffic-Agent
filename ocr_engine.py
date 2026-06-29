import cv2
import numpy as np
import paddle.inference as paddle_infer
import os

# 你的模型文件路径（确保文件夹里有 inference.pdmodel 和 inference.pdiparams）
MODEL_DIR = '/wangyuanzhu/my_car_plate_project/models/ch_PP-OCRv4_rec_infer'

def load_predictor(model_dir):
    model_file = os.path.join(model_dir, 'inference.pdmodel')
    params_file = os.path.join(model_dir, 'inference.pdiparams')
    
    # 配置推理引擎 (核心：绕过所有高级封装)
    config = paddle_infer.Config(model_file, params_file)
    config.disable_gpu() # 离线环境下强制 CPU 推理
    config.switch_use_feed_fetch_ops(False)
    
    # 创建预测器
    predictor = paddle_infer.create_predictor(config)
    return predictor

# 初始化引擎
try:
    print("[系统] 正在尝试直接加载模型引擎...")
    predictor = load_predictor(MODEL_DIR)
    print("[成功] 核心推理引擎加载完毕，无需任何依赖包！")
except Exception as e:
    print(f"[致命错误] 即使是底层引擎也无法加载: {e}")
    print("这说明你的环境缺少 PaddlePaddle 核心库，或者模型路径不对。")