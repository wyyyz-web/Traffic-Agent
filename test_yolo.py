import cv2
import os
from ultralytics import YOLO

# ==========================================
# 1. 加载检测模型（换上你训练的满分权重）
# ==========================================
model_path = '/wangyuanzhu/my_car_plate_project/runs/detect/train/weights/best.pt'
if not os.path.exists(model_path):
    print(f"未找到专属模型，请检查路径: {model_path}")
    exit()

model = YOLO(model_path)

# ==========================================
# 2. 读取要测试的植物遮挡图
# ==========================================
test_image_path = '/wangyuanzhu/my_car_plate_project/robust_test/plant_cover.jpg'

# 必须先用 OpenCV 读取图片放入内存
img = cv2.imread(test_image_path)

if img is None:
    print(f" [错误] 无法读取图片，请检查路径: {test_image_path}")
else:
    print("开始进行车牌定位检测...")
    
    # 3. 进行推理
    results = model(test_image_path)

    # 确保输出目录存在
    output_dir = '/wangyuanzhu/my_car_plate_project/chars_output/'
    os.makedirs(output_dir, exist_ok=True)

    # 检查是否检测到车牌
    if len(results[0].boxes) == 0:
        print(" 失败！YOLO 没有在这张照片里找到任何车牌（可能是遮挡太严重了）。")
    else:
        print(f" 成功！模型找到了 {len(results[0].boxes)} 个车牌目标。")

        # ==========================================
        # 🌟 核心需求：生成并保存可视化定位图 (带画框)
        # ==========================================
        # plot() 方法会自动在原图上画好预测框和置信度
        annotated_img = results[0].plot() 
        
        localization_save_path = os.path.join(output_dir, 'localization_result.jpg')
        cv2.imwrite(localization_save_path, annotated_img)
        print(f" -> 整体定位图已保存至: {localization_save_path} (请打开此图查看画框效果！)")

        # ==========================================
        # 保留原逻辑：裁剪纯车牌供 OCR 使用
        # ==========================================
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # --- 动态 Padding 留白防边缘截断 ---
            w = x2 - x1
            h = y2 - y1
            x1 = max(0, int(x1 - w * 0.03))
            x2 = min(img.shape[1], int(x2 + w * 0.03))
            y1 = max(0, int(y1 - h * 0.02))
            y2 = min(img.shape[0], int(y2 + h * 0.02))
            # ----------------------------------------
            
            crop = img[y1:y2, x1:x2]
            
            crop_save_path = os.path.join(output_dir, 'debug_boxes.jpg')
            cv2.imwrite(crop_save_path, crop)
            print(f" -> 裁剪的车牌图已保存至: {crop_save_path}")