from yolov5 import train

train.run(
    data='dataset/fishing.yaml',
    imgsz=640,
    batch=16,
    epochs=100,
    weights='yolov5s.pt',  # o yolov5n.pt para más ligero
    name='fishing_detector'
)