import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model_t = YOLO("D:/supervised/ultralytics-main-服务器 - 副本/ultralytics-main/runs/detect/train55/weights/best.pt")  # 此处填写教师模型的权重文件地址

    model_t.model.model[-1].set_Distillation = True  # 不用理会此处用于设置模型蒸馏

    model_s = YOLO(r'yolov8.yaml')  # 学生文件的yaml文件 or 权重文件地址

    model_s.train(data='rolled1.yaml',
                  # 将data后面替换你自己的数据集地址
                  cache=False,
                  imgsz=640,
                  epochs=200,
                  single_cls=False,  # 是否是单类别检测
                  batch=1,
                  close_mosaic=10,
                  workers=0,
                  device='0',
                  optimizer='SGD',  # using SGD
                  amp=True,  # 如果出现训练损失为Nan可以关闭amp
                  project='runs/train',
                  name='exp',
                  model_t=model_t.model
                  )