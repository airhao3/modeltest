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
YOLO 可以处理不同类型的输入源进行推理，如下表所示。输入源包括静态图像、视频流和各种数据格式。表格还指明了每个源是否可以在流模式下使用，即带有参数 stream=True ✅。流模式对于处理视频或实时流非常有利，因为它会创建结果生成器，而不是将所有帧加载到内存中。

处理长视频或大型数据集时，使用 stream=True 可以有效管理内存。当 stream=False 时，所有帧或数据点的结果都会存储在内存中，这对于大型输入来说可能会迅速累积并导致内存不足错误。相比之下，stream=True 使用生成器，只将当前帧或数据点的结果保留在内存中，显著减少内存消耗并防止内存不足问题。

| 来源            | 参数                                     | 类型           | 备注                                                                                   |
| ------------- | --------------------------------------- | -------------- | ------------------------------------------------------------------------------------ |
| 图像           | 'image.jpg'                             | str 或 Path    | 单个图片文件。                                                                         |
| URL           | 'https://ultralytics.com/images/bus.jpg'| str            | 图片的 URL。                                                                          |
| 截屏           | 'screen'                                | str            | 捕获屏幕截图。                                                                          |
| PIL           | Image.open('im.jpg')                    | PIL.Image      | HWC 格式，RGB 通道。                                                                   |
| OpenCV        | cv2.imread('im.jpg')                    | np.ndarray     | HWC 格式，BGR 通道，uint8 (0-255)。                                                   |
| numpy         | np.zeros((640,1280,3))                  | np.ndarray     | HWC 格式，BGR 通道，uint8 (0-255)。                                                   |
| torch         | torch.zeros(16,3,320,640)               | torch.Tensor   | BCHW 格式，RGB 通道，float32 (0.0-1.0)。                                               |
| CSV           | 'sources.csv'                           | str 或 Path    | 包含图像、视频或目录路径的 CSV 文件。                                                   |
| 视频 ✅        | 'video.mp4'                             | str 或 Path    | 视频文件，格式如 MP4、AVI 等。                                                         |
| 目录 ✅        | 'path/'                                 | str 或 Path    | 包含图像或视频的目录路径。                                                             |
| glob ✅        | 'path/*.jpg'                            | str            | 匹配多个文件的 glob 模式。使用 * 字符作为通配符。                                       |
| YouTube ✅     | 'https://youtu.be/LNwODJXcvt4'          | str            | YouTube 视频的 URL。                                                                  |
| 流媒体 ✅       | 'rtsp://example.com/media.mp4'          | str            | 流媒体协议的 URL，如 RTSP、RTMP、TCP 或 IP 地址。                                       |
| 多流媒体 ✅     | 'list.streams'                          | str 或 Path    | *.streams 文本文件，每行一个流媒体 URL，即 8 个流将以批量大小 8 运行。                   |

