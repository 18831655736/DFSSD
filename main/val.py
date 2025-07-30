from ultralytics import YOLO
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if __name__ == '__main__':


 model = YOLO("D:/supervised/ultralytics-main-服务器/ultralytics-main/runs/detect/train59/weights/best.pt")  # 用于迁移训练的权重文件路径

 results = model.val(split='val',data="combined_data.yaml",save=True)