import os
import sqlite3
import json
import logging
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
from data_pipeline import serialize_to_json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_PATH = "traffic_agent.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            timestamp DATETIME,
            camera_id TEXT,
            license_plate TEXT,
            confidence REAL,
            bbox TEXT,
            raw_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

def process_batch(image_dir="./robust_test", weight_path="./runs/detect/train/weights/best.pt"):
    logging.info("Initializing YOLO model and PaddleOCR engine...")
    try:
        yolo_model = YOLO(weight_path)
        ocr_engine = PaddleOCR(use_textline_orientation=True, lang="ch")
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    valid_extensions = (".jpg", ".jpeg", ".png")
    if not os.path.exists(image_dir):
        logging.error(f"Directory not found: {image_dir}")
        return
        
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        logging.warning(f"No valid images found in {image_dir}")
        return

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        logging.info(f"Processing image: {img_name}")
        
        results = yolo_model(img_path, verbose=False)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bbox_list = [x1, y1, x2, y2]
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                plate_crop = img[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue
                    
                ocr_results = ocr_engine.ocr(plate_crop, cls=True)
                
                if not ocr_results or not ocr_results[0]:
                    continue
                    
                for line in ocr_results[0]:
                    plate_text = line[1][0]
                    confidence_score = float(line[1][1])
                    
                    json_str = serialize_to_json(plate_text, confidence_score, bbox_list)
                    data_dict = json.loads(json_str)
                    
                    try:
                        cursor.execute('''
                            INSERT INTO vehicle_records (event_id, timestamp, camera_id, license_plate, confidence, bbox, raw_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            data_dict["event_id"],
                            data_dict["timestamp"],
                            data_dict["location_metadata"]["camera_id"],
                            data_dict["detection_data"]["license_plate"],
                            data_dict["detection_data"]["confidence"],
                            json.dumps(data_dict["detection_data"]["bounding_box"]),
                            json_str
                        ))
                        logging.info(f"Database insertion successful - Plate: {plate_text} from {img_name}")
                    except sqlite3.IntegrityError:
                        logging.warning(f"Duplicate event_id detected for {plate_text}")

    conn.commit()
    conn.close()
    logging.info("End-to-end inference and data persistence completed.")

if __name__ == "__main__":
    init_db()
    process_batch()
