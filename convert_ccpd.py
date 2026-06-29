import os
import cv2

IMG_DIR = './datasets/images/train'
LABEL_DIR = './datasets/labels/train'

def convert_ccpd_to_yolo(filename):
    try:
        parts = filename.split('-')
        coord_str = parts[2]
        points = [[int(p.split('&')[0]), int(p.split('&')[1])] for p in coord_str.split('_')]
        img_w, img_h = 720, 1160
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        width = xmax - xmin
        height = ymax - ymin
        x_center = (xmin + width / 2) / img_w
        y_center = (ymin + height / 2) / img_h
        norm_width = width / img_w
        norm_height = height / img_h
        return f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
    except:
        return None

if __name__ == "__main__":
    os.makedirs(LABEL_DIR, exist_ok=True)
    image_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    for img_name in image_files:
        label = convert_ccpd_to_yolo(img_name)
        if label:
            with open(os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + '.txt'), 'w') as f:
                f.write(label)
    print("转换完成！")
