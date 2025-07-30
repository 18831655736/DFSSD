import warnings

import torch

from ultralytics import YOLO
from ultralytics.utils.metrics import PseudoLabeling  # 引用伪标签模块

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # 初始化教师模型
    model_t = YOLO("path_to_teacher_model")  # 教师模型权重文件地址
    model_t.model.model[-1].set_Distillation = True  # 设置模型蒸馏
    model_s = YOLO(r'yolov8.yaml')  # 学生模型配置文件地址

    pseudo_labeler = PseudoLabeling(ema_decay=0.999, initial_threshold=0.5)

    # 训练配置
    model_s.train(
        data='',
        cache=False,
        imgsz=640,
        epochs=100,
        single_cls=False,
        batch=1,
        close_mosaic=10,
        workers=0,
        device='0',
        optimizer='SGD',
        amp=True,
        project='runs/train',
        name='exp',
        model_t=model_t.model,
    )


    for epoch in range():
        pseudo_labels = torch.tensor([])
        confidences = torch.tensor([])
        labels = torch.tensor([])
        pseudo_labeler.update_ema(confidences, labels)
        filtered_pseudo_labels, filtered_confidences, filtered_labels = pseudo_labeler.filter_pseudo_labels(
            pseudo_labels, confidences, labels)