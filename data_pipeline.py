import json
import datetime
import uuid

def serialize_to_json(plate_text, ocr_confidence, bbox_coords, camera_id="CAM_N_001"):
    current_time = datetime.datetime.now().isoformat()
    unique_hash = str(uuid.uuid4())[:8]
    
    traffic_record = {
        "event_id": f"{camera_id}_{int(datetime.datetime.now().timestamp())}_{unique_hash}",
        "timestamp": current_time,
        "location_metadata": {
            "camera_id": camera_id,
            "intersection": "Main_St_and_1st_Ave"
        },
        "detection_data": {
            "license_plate": plate_text,
            "confidence": float(ocr_confidence),
            "bounding_box": bbox_coords
        }
    }
    
    json_output = json.dumps(traffic_record, indent=4, ensure_ascii=False)
    return json_output
