# 🚀 FLDetn - Ultralytics-based Aerial Object Detection

An **Ultralytics YOLOv8** implementation of the paper:  
🔗 **[FLDet: Faster and Lighter Aerial Object Detector](https://ieeexplore.ieee.org/document/10798479)**  
:octocat: [Official Paper GitHub](https://github.com/wsy-yjys/FLDet/tree/main)



## 📌 Project Highlights

- ✅ Based on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- ✅ Modular design for integrating custom blocks (IEPR, ECM, CSNeck3in)
- ✅ Supports YAML-based model configuration
- ✅ Easy-to-extend for aerial datasets or low-latency applications



## 🧱 Architecture

```txt
YOLOv8 backbone
└── Conv
└── IEPR (Improved Efficient Pyramid Residual)
└── ECM (Efficient Context Module)
└── CSNeck3in
└── Detect Head
```

All custom modules are implemented under `models/FLDetn/ultralytics/nn/modules`.



## 🧪 Dataset & Training

	•	🛰️ Compatible with aerial object detection datasets
	•	📁 Follows standard YOLO format:
 ```txt
├── images/
│   ├── train/
│   ├── val/
└── labels/
    ├── train/
    ├── val/
```

### 🏋️‍♀️ Training example:
```txt
yolo task=detect mode=train \
     model=models/FLDetn/FLDet-n.yaml \
     data=path/to/data.yaml \
     epochs=100 imgsz=640
```



## 📁 File Structure
```txt
models/
├── FLDetn/
│   ├── FLDet-n.yaml                  # Model configuration (YAML)
│   ├── ultralytics/
│   │   └── nn/
│   │       └── modules/
│   │           ├── iepr.py          # IEPR module
│   │           ├── ecm.py           # ECM module
│   │           └── csneck3in.py     # CSNeck3in module
│   ├── safe_utils.py                # PyTorch 2.6+ safe loading patch
```


## 📊 Results (working...)

| Model         | Params | FPS | AP50 |
|---------------|--------|-----|------|
| FLDet-n       | 2.9M   | ... |  ... |
| FLDet-n + ECM | 3.1M   | ... |  ... |

> 🔍 * Will beTested on internal aerial object dataset.*

 

## 🛠️ Compatibility Notes

⚠️ As of PyTorch 2.6, the default torch.load() behavior uses weights_only=True.
This will fail if your .pt file contains custom modules (like ours).

✅ To fix this, use the patched loading function:
```shell
from safe_utils import YOLO_safe

model = YOLO_safe("runs/detect/train/weights/best.pt")
```
This registers custom modules to torch.serialization.safe_globals before loading.


## 🤝 Acknowledgements

  •💡 Paper: [FLDet: Faster and Lighter Aerial Object Detector](https://ieeexplore.ieee.org/document/10798479) (2022)     
  • :octocat: Offifcial Github: https://github.com/wsy-yjys/FLDet/tree/main      
	•	🛠️ Framework: Ultralytics YOLOv8      
	•	✍️ Authors: wsy-yjys       


## 📬 Contact

Have suggestions or questions?
Please feel free to open an issue or pull request!

### ⭐️ If you found this repository useful, please consider giving it a star!


