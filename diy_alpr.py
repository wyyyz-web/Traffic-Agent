import cv2
import os
import numpy as np
from ultralytics import YOLO

# 基础路径配置
BASE_DIR = '/wangyuanzhu/my_car_plate_project'
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'runs/detect/train/weights/best.pt')
TEST_IMG_PATH = os.path.join(BASE_DIR, 'test_car.jpg')
CHARS_DIR = os.path.join(BASE_DIR, 'chars_output')

os.makedirs(CHARS_DIR, exist_ok=True)

if not os.path.exists(YOLO_MODEL_PATH):
    YOLO_MODEL_PATH = 'yolov8n.pt'

model = YOLO(YOLO_MODEL_PATH)

def generate_char_templates():
    """动态绘制标准字典 (黑底白字)"""
    templates = {}
    alphabet = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    for char in alphabet:
        canvas = np.zeros((60, 40), dtype=np.uint8)
        cv2.putText(canvas, char, (5, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 4, cv2.LINE_AA)
        templates[char] = cv2.resize(canvas, (20, 40))
    return templates

def recognize_char(char_img, templates):
    """像素级模板匹配"""
    best_score = -1
    best_char = "?"
    for char, template in templates.items():
        res = cv2.matchTemplate(char_img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_char = char
    return best_char, best_score

def run_full_pipeline():
    if not os.path.exists(TEST_IMG_PATH):
        print(f"[错误] 找不到测试图片: {TEST_IMG_PATH}")
        return

    print("[系统] 正在动态构建本地离线字形字典...")
    templates = generate_char_templates()

    img = cv2.imread(TEST_IMG_PATH)
    print("[阶段一] YOLO 定位车牌中...")
    results = model.predict(source=img, conf=0.4, save=False, verbose=False)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            plate_img = img[max(0, y1-5):min(img.shape[0], y2+5), 
                            max(0, x1-5):min(img.shape[1], x2+5)]
            
            # 图像预处理
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 【核心升级：智能色彩翻转机制】
            # 如果整张图白色像素过半，说明背景是白的、字是黑的，这不符合OpenCV轮廓和字典要求，必须强行反转
            if np.mean(binary) > 127:
                print("[调试提示] 检测到车牌颜色反转，正在自动调整为黑底白字...")
                binary = cv2.bitwise_not(binary)
            
            # 保存最终决定用的黑白图
            cv2.imwrite(os.path.join(CHARS_DIR, "debug_binary_final.jpg"), binary)
            
            # 轮廓查找
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            char_boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                # 收集所有合理的块
                if 0.05 < aspect_ratio < 2.5 and h > 10 and w > 2:
                    char_boxes.append((x, y, w, h))
            
            char_boxes = sorted(char_boxes, key=lambda b: b[0])
            print(f"[阶段二] 字符切片扫描完毕，共发现 {len(char_boxes)} 个可能的目标图形。")
            print("================ 开始打印底层的匹配数据 ================")
            
            final_plate_text = []
            for i, (x, y, w, h) in enumerate(char_boxes):
                char_crop = binary[y:y+h, x:x+w]
                char_resized = cv2.resize(char_crop, (20, 40))
                
                pred_char, score = recognize_char(char_resized, templates)
                
                # 调试输出：不论分数多低，都把它打印出来！
                print(f" -> 图形位置 {i+1} (x={x}): 最像 【{pred_char}】 (相似度分数: {score:.3f})")
                
                # 只要分数不是差得离谱（大于0.1），就放进最终结果里看看
                if score > 0.1:
                    final_plate_text.append(pred_char)
            
            result_str = "".join(final_plate_text)
            print("======================================================")
            print(f"✨【最新离线识别尝试】车牌号码可能为: {result_str}")
            print("======================================================\n")
            return

if __name__ == "__main__":
    run_full_pipeline()
