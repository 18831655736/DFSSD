from ultralytics import YOLO
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if __name__ == '__main__':

     # 训练数据集：
    # model = YOLO('yolov8n.pt')  # 如果要训练如pose，该对应的yaml和权重即可
    model = YOLO('yolov8s.pt')
    model = YOLO('yolov8.yaml')
    results = model.train(data='rolled.yaml',epochs=200)

    #metrics = model.val()  # 在验证集上评估模型性能
    # 预测结果
    # model = YOLO('yolov8n.pt') #常用模型yolov8n-seg.pt、yolov8n.pt、yolov8n-pose.pt
    # model.predict('ultralytics/assets', save=True) #测试图片文件夹，并且设置保存True

    # 如果中断后，可以改为以下代码：
    # model = YOLO('last.pt')  # last.pt文件的路径
    # results = model.train(resume=True)