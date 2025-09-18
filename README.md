

#  Mask Detect and Counting

![demo_video](https://github.com/user-attachments/assets/cab03f4a-ee9b-4aff-93ed-020c6d6afeec)

https://www.youtube.com/watch?v=dz6mcj71wLc

A real-time mask detection and counting system based on YOLOv8s that identifies mask-wearing status in live CCTV footage and counts the total number of people and mask-wearing individuals through object tracking.


##  project structure

```
Mask-detect-and-counting/
├── mask_checking/         # Training results and model files
├── data.yaml              # Data path and class name configuration
├── hyp_custom.yaml        # YOLOv8s training configuration
├── main.ipynb             # YOLOv8s fine-tuning Jupyter notebook
├── mask_counting.py       # Main detection & counting
├── yolov8s.pt             # pretrained model
└── README.md
```


## Main Functions
- YOLOv8s training in Jupyter notebook
- Detecting masks, assigning identity IDs, and tracking
- Counting masked and non-masked people

## Training Result
<img width="2400" height="1200" alt="image" src="https://github.com/user-attachments/assets/779f37ef-a020-46ed-8b98-e9b919d08365" />

| Metric | Value | Description |
|--------|-------|-------------|
| Precision | 88.89% | Accuracy of positive predictions |
| Recall | 87.21% | Ability to find all positive instances |
| mAP50 | 91.73% | Mean Average Precision at IoU 0.5 |
| mAP50-95 | 59.71% | Mean Average Precision at IoU 0.5-0.95 |

*Results are from the final epoch (epoch 100) with best model weights.*

##  Model infomation

* Model: YOLOv8s (Ultralytics)
* Classes:
  * `0`: Face with mask
  * `1`: Face without mask
* Training Data: OpenCV mask image dataset
* Framework: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

## Limitations
- Object detection fails when object size is too small, resulting in inaccurate counting
- Individual tracking is based on bounding box positions, making it unable to distinguish between people within the same bounding box

---

## 📎 Reference Libraries

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [abewley/sort](https://github.com/abewley/sort)


