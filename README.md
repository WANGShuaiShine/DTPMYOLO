# Pedestrian mask-wearing tracking using the improved Yolov5s-6.0 & DeepSORT

## Introduction

This repository features a two-stage tracking system. First, pedestrian mask-wearing detections are generated using the Improved [YOLOv5](https://github.com/ultralytics/yolov5). These detections are then passed to the [DeepSort](https://github.com/ZQPei/deep_sort_pytorch)  for mask-wearing tracking.

The project combined mask detection with pedestrian detection and tracking, proposing an approach for tracking pedestrians wearing masks. This algorithm enables continuous and stable tracking of pedestrians wearing masks, suitable for applications in epidemic prevention and control.

## Tutorials

* [Yolov5 training on Custom Data (link to external repository)](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp;
* [DeepSORT training (link to external repository)](https://github.com/ZQPei/deep_sort_pytorch#training-the-re-id-model)&nbsp;

## Installing PyTorch (GPU and CPU versions)

DTPM_YOLO can be used on both CPU and GPU. However, training on a CPU is very time-consuming and labor-intensive, so it is recommended to install the GPU version of PyTorch if possible. For those who cannot, it is best to rent a server.

### When installing the GPU version, please note the following points:

1. Before installation, be sure to update your graphics card drivers by downloading and installing the appropriate drivers from the official website.
2. For 30-series graphics cards, only the CUDA 11 version can be used.
3. It is essential to create a virtual environment to prevent conflicts between different deep learning frameworks.




## How to use the custom code

1. Clone the repository recursively:

`git clone --recurse-submodules https://github.com/WANGShuaiShine/DTPM_YOLO.git`

If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/blob/master/requirements.txt) dependencies installed, including torch>=1.7. To install, run:

`pip install -r requirements.txt`


## Run the tracking sources

Tracking can be run on most video formats

```bash
python3 pedestrain_mask_track.py --source ... --show-vid  # show live inference results as well
```

- Video:  `--source yourvediofilename.mp4`
- Webcam:  `--source 0`


