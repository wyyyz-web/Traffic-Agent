import cv2
import numpy as np
import paddle.inference as paddle_infer
import os
import math

# --- 1. 配置路径 ---
BASE_DIR = '/wangyuanzhu/my_car_plate_project'
MODEL_DIR = os.path.join(BASE_DIR, 'models/ch_PP-OCRv4_rec_infer')
DICT_PATH = os.path.join(MODEL_DIR, 'ppocr_keys_v1.txt')
# 【关键】直接指定我们要识别的正确文件
IMAGE_PATH = os.path.join(BASE_DIR, 'chars_output/debug_boxes.jpg')

class PaddleRecEngine:
    def __init__(self, model_dir):
        model_file = os.path.join(model_dir, 'inference.pdmodel')
        params_file = os.path.join(model_dir, 'inference.pdiparams')
        
        config = paddle_infer.Config(model_file, params_file)
        config.disable_gpu()
        config.switch_use_feed_fetch_ops(False)
        
        self.predictor = paddle_infer.create_predictor(config)
        
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            self.labels = [line.strip() for line in f.readlines()]
            self.labels.append(" ") 

    def preprocess(self, img):
        import math
        import cv2
        import numpy as np

        h, w = img.shape[:2]
        
        # ==========================================
        # 物理抗畸变：横向拉伸算法 (Anti-Perspective Stretch)
        # 因为侧拍导致字变窄了，我们强行把图片的宽度放大 1.3 到 1.5 倍
        # 把被挤压的“浙”字给重新拉宽，暴露出三点水的特征！
        # ==========================================
        stretch_ratio = 1.4  # 你可以根据侧拍的严重程度调整这个值 (1.2 ~ 1.5)
        img = cv2.resize(img, (int(w * stretch_ratio), h))
        
        # 更新拉伸后的宽高
        h, w = img.shape[:2]

        # ... 下面保留你原来的等比例缩放和黑底/白底填充代码 ...
        img_H, img_W = 48, 320
        # ...
        h, w = img.shape[:2]

        # 1. 严格等比例计算宽度
        ratio = w / float(h)
        if math.ceil(img_H * ratio) > img_W:
            resized_w = img_W
        else:
            resized_w = int(math.ceil(img_H * ratio))

        # 2. 缩放原图 (保留所有暗部细节和色彩)
        resized_image = cv2.resize(img, (resized_w, img_H))

        # 3. 创建纯黑背景板并贴图
        padding_im = np.zeros((img_H, img_W, 3), dtype=np.uint8)
        padding_im[:, 0:resized_w, :] = resized_image

        # 4. 颜色转换、归一化并传给模型
        padding_im = cv2.cvtColor(padding_im, cv2.COLOR_BGR2RGB)
        padding_im = padding_im.astype('float32') / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        padding_im = (padding_im - mean) / std
        padding_im = padding_im.transpose((2, 0, 1))

        return padding_im[np.newaxis, :]
    
    def predict(self, img):
        input_data = self.preprocess(img)
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        input_handle.reshape(input_data.shape)
        input_handle.copy_from_cpu(input_data)
        self.predictor.run()
        
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        return output_handle.copy_to_cpu()

    def decode(self, result):
        pred_idx = np.argmax(result[0], axis=1)
        char_list = []
        last_idx = 0
        for idx in pred_idx:
            if idx > 0 and idx != last_idx:
                if int(idx) - 1 < len(self.labels):
                    char_list.append(self.labels[int(idx) - 1])
            last_idx = idx
            
        # 1. 拿到原始输出，第一步先清理掉所有空格！
        # 把 "A BS9676" 变成 "ABS9676"
        raw_text = "".join(char_list).replace(" ", "")

        # ==========================================
        # 💡 商业级最终防线 V2.0：后缀提取法
        # ==========================================
        import re
        
        # 我们不再强求开头是汉字，而是直接去字符串的【末尾】寻找合法的车牌主体！
        # 规则：最后部分必须是 "1个大写字母 + 5或6位大写字母或数字"
        body_pattern = r'([A-Z][0-9A-Z]{5,6})$'
        
        match = re.search(body_pattern, raw_text)
        
        if match:
            # group(1) 拿到的就是车牌的后半截，比如 "ABS9676"
            rest = match.group(1) 
            
            HOME_PROVINCE = "浙"
            
            # 直接暴力拼接！无论前面是 "H", "A", 还是什么都没，全部换成 "浙"
            corrected_text = HOME_PROVINCE + rest
            
            # 【修改这里】：把原来带有括号的 return 删掉，直接返回干净的结果！
            return corrected_text
                
        # 如果连车牌后半截都认不出来（格式彻底坏了），那只能返回原样
        return raw_text
    
if __name__ == "__main__":
    print("[系统] 正在初始化引擎...")
    engine = PaddleRecEngine(MODEL_DIR)
    
    # 指向你的那张带有树枝的原始裁剪图（注意用彩色原图，不要用之前的二值化图）
    IMAGE_PATH = '/wangyuanzhu/my_car_plate_project/chars_output/debug_boxes.jpg' 
    img = cv2.imread(IMAGE_PATH)
    
    if img is None:
        print(f" [错误] 无法找到图片: {IMAGE_PATH}")
    else:
        print(f" [成功] 已读取图片，开始 TTA 多视角推理...\n")
        
        h, w = img.shape[:2]
        
        # 视角 1：原汁原味的原图
        view1 = img.copy()
        
        # 视角 2：切掉顶部 25% (试图避开挡在上面的树枝)，强行拉伸
        view2 = img[int(h*0.25):h, 0:w].copy()
        
        # 视角 3：切掉底部 25% (试图避开挡在下面的树枝)，强行拉伸
        view3 = img[0:int(h*0.75), 0:w].copy()
        
        # 视角 4：CLAHE 增强对比度版 (提亮暗部)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        view4 = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        views = {
            "1. 原始图像": view1,
            "2. 削顶视图": view2,
            "3. 削底视图": view3,
            "4. 对比度强化": view4
        }
        
        print("================= AI 的四次疯狂猜测 =================")
        for name, view_img in views.items():
            result = engine.predict(view_img)
            text = engine.decode(result)
            print(f"[{name}] 模型看到的是: 【 {text} 】")
        print("=====================================================")
        
        print("\n 总结：你可以观察这四个结果。如果它们分别猜对了一部分（比如削顶视图猜对了底部的数字，削底视图猜对了顶部的字母），在真实的商业代码中，我们就可以写一套正则匹配逻辑，把它们‘拼’成一个完美的车牌号！")