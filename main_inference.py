import os
import sqlite3
import json
import logging
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
    logging.info("SQLite database and schema initialized successfully.")

def process_batch(image_dir="./test_images"):
    # TODO: Load actual YOLO26n best.pt and PaddleOCR engine here
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Mocking inference results for pipeline validation
    mock_detections = [
        {"plate": "SU-A88888", "conf": 0.99, "box": [10, 20, 100, 50]},
        {"plate": "SU-B12345", "conf": 0.95, "box": [15, 25, 110, 55]}
    ]

    for det in mock_detections:
        json_str = serialize_to_json(det["plate"], det["conf"], det["box"])
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
            logging.info(f"Database insertion successful for plate: {det['plate']}")
        except sqlite3.IntegrityError:
            logging.warning(f"Duplicate event_id detected: {data_dict['event_id']}")

    conn.commit()
    conn.close()
    logging.info("Batch processing pipeline execution completed.")

if __name__ == "__main__":
    init_db()
    process_batch()
