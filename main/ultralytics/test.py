from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.pt")  # load an official model
model = YOLO("D:/supervised/ultralytics-main-服务器/ultralytics-main/runs/detect/train55/weights/best.pt")  # load a custom model

# Predict with the model
results = model("C:/Users/13910/Desktop/新建文件夹/combined_image1.jpg",save=True, save_conf=True, save_txt=True, name='output')  # predict on an image
