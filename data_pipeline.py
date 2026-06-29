import json
import datetime

def serialize_to_json(plate_text, ocr_confidence, bbox_coords, camera_id="CAM_N_001"):
    # 获取当前时间戳
    current_time = datetime.datetime.now().isoformat()
    
    # 构建结构化的时空交通数据字典
    traffic_record = {
        "event_id": f"{camera_id}_{int(datetime.datetime.now().timestamp())}",
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
    
    # 转化为标准的 JSON 格式 (ensure_ascii=False 保证中文字符正常显示)
    json_output = json.dumps(traffic_record, indent=4, ensure_ascii=False)
    return json_output

if __name__ == "__main__":
    # 模拟 YOLO 和 OCR 吐出的识别结果
    mock_plate = "苏A88888"
    mock_conf = 0.985
    mock_bbox = [120, 300, 250, 350] 
    
    print("正在测试数据流水线 JSON 序列化...\n")
    result = serialize_to_json(mock_plate, mock_conf, mock_bbox)
    print(result)
