## Model Test Pages


### YOLO v8模型使用文档

#### YOLO 模型predict模式：
- 多数据源并行
- 数据流模式
- 批量处理
- 友好整合

YOLO 模型在推理过程中会返回两种结果：一种是包含 Results 对象的 Python 列表，另一种是在使用 stream=True 参数时返回的内存高效的 Results 对象生成器。


```
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["im1.jpg", "im2.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk

```
