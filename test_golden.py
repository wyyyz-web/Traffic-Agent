import cv2
import numpy as np
import paddle.inference as paddle_infer
import os

MODEL_DIR = '/wangyuanzhu/my_car_plate_project/models/ch_PP-OCRv4_rec_infer'
DICT_PATH = os.path.join(MODEL_DIR, 'ppocr_keys_v1.txt')

class MiniEngine:
    def __init__(self):
        config = paddle_infer.Config(os.path.join(MODEL_DIR, 'inference.pdmodel'),
                                     os.path.join(MODEL_DIR, 'inference.pdiparams'))
        config.disable_gpu()
        self.predictor = paddle_infer.create_predictor(config)
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        new_w = max(320, int(48 * (w / h)))
        img = cv2.resize(img, (new_w, 48))
        img = img.astype('float32') / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose((2, 0, 1))
        return img[np.newaxis, :]

    def decode(self, pred_idx):
        char_list = []
        last_idx = 0
        for idx in pred_idx:
            if idx > 0 and idx != last_idx:
                if int(idx) - 1 < len(self.labels):
                    char_list.append(self.labels[int(idx) - 1])
            last_idx = idx
        return "".join(char_list)

if __name__ == "__main__":
    print("[系统] 正在初始化引擎...")
    engine = MiniEngine()
    
    print("[系统] 正在生成完美测试图...")
    img = np.full((48, 320, 3), 255, dtype=np.uint8)
    cv2.putText(img, "8888", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    input_data = engine.preprocess(img)
    input_names = engine.predictor.get_input_names()
    handle = engine.predictor.get_input_handle(input_names[0])
    handle.reshape(input_data.shape)
    handle.copy_from_cpu(input_data)
    engine.predictor.run()
    
    output_names = engine.predictor.get_output_names()
    out_handle = engine.predictor.get_output_handle(output_names[0])
    result = out_handle.copy_to_cpu()
    
    max_probs = np.max(result[0], axis=1)[:10]
    print(f"DEBUG: 完美图前10个置信度: {max_probs}")
    
    all_indices = np.argmax(result[0], axis=1)
    print(f"DEBUG: 完美图所有索引: {all_indices}")
    print(f"\n==============================")
    print(f"完美图识别结果: 【 {engine.decode(all_indices)} 】")
    print(f"==============================\n")