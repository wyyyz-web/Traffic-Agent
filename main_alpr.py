import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os

# 1. 初始化模型
model_path = '/wangyuanzhu/my_car_plate_project/runs/detect/train/weights/best.pt'
if not os.path.exists(model_path):
    model_path = 'yolov8n.pt'

model = YOLO(model_path)
ocr = PaddleOCR(use_textline_orientation=True, lang='ch')

def recognize_plate(image_path):
    print(f"正在处理图片: {image_path}")
    results = model.predict(source=image_path, conf=0.5, save=False, verbose=False)
    img = cv2.imread(image_path)
    
    found_plate = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            found_plate = True
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            plate_img = img[y1:y2, x1:x2]
            temp_crop_path = "temp_plate.jpg"
            cv2.imwrite(temp_crop_path, plate_img)
            
            ocr_result = ocr.ocr(temp_crop_path, cls=True)
            
            if ocr_result and ocr_result[0]:
                for line in ocr_result[0]:
                    print(f"识别到的车牌号: {line[1][0]} (置信度: {line[1][1]:.2f})")
            else:
                print("未能在车牌区域识别出文字")
                
    if not found_plate:
        print("YOLO 未能在图中检测到车牌。")

if __name__ == "__main__":
    # 【关键修改】指向植物遮挡的测试图
    test_img = '/wangyuanzhu/my_car_plate_project/robust_test/plant_cover.jpg' 
    if os.path.exists(test_img):
        recognize_plate(test_img)
    else:
        print(f" [错误] 请确保图片已上传到: {test_img}")