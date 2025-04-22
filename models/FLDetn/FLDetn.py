import os
import yaml
import random
import numpy as np
import pandas as pd
from datetime import datetime
import torch

# --- sys.path 설정 추가 ---
import sys
from pathlib import Path

# 1. 설치된 ultralytics 모듈이 import되어 있으면 제거 (우선순위 충돌 방지)
if 'ultralytics' in sys.modules:
    del sys.modules['ultralytics']

# 2. 현재 파일 기준으로 FLDetn 내부 ultralytics 경로 등록
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path().resolve()

# submission_1_*.py → BASE_DIR = .
# FLDetn.py → BASE_DIR = models/FLDetn → 한 단계 위로
if (BASE_DIR / "ultralytics").exists():
    ULTRA_PATH = BASE_DIR
else:
    ULTRA_PATH = BASE_DIR / "models" / "FLDetn"

sys.path.insert(0, str(ULTRA_PATH))  # 우선순위 가장 높게 등록

# 3. 이제 import 가능
import safe_utils
from safe_utils import YOLO_safe, setup_safe_globals, SafeTrainer
# from ultralytics.engine.trainer import BaseTrainer
from ultralytics import settings
settings.update({'datasets_dir': './'})
    

def train_model(ex_dict, model, model_yaml_path):

    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    name = f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    task = f"{ex_dict['Experiment Time']}_Train"


    ex_dict['Train Results'] =model.train(
        # model = model_yaml_path, # f"{ex_dict['Model Name']}.yaml",
        name=name,
        data=ex_dict['Data Config'] ,
        epochs=ex_dict['Epochs'],
        imgsz=ex_dict['Image Size'],
        batch=ex_dict['Batch Size'],
        patience=20,
        save=True,
        device=ex_dict['Device'],
        exist_ok=True,
        verbose=False,
        optimizer=ex_dict['Optimizer'],
        lr0=ex_dict['LR'],  
        weight_decay = ex_dict['Weight Decay'],
        momentum = ex_dict['Momentum'],
        pretrained=False,
        amp=False,
        task = task,
        # cfg=None,
        project =f"{ex_dict['Output Dir']}",
        # trainer =SafeTrainer(overrides=overrides)
    )
    pt_path = f"{ex_dict['Output Dir']}/{name}/weights/best.pt"
    ex_dict['PT path'] = pt_path
    ex_dict['Model'].load(pt_path)
    return ex_dict

def detect_and_save_bboxes(model_path, image_paths):

    model = YOLO_safe(model_path)
    # model = YOLO(model_path)
    
    results_dict = {}

    for img_path in image_paths:
        results = model(img_path, verbose=False, task='detect')
        img_results = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                bbox = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                img_results.append({
                    'bbox': bbox,  # [x1, y1, x2, y2]
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                })
        results_dict[img_path] = img_results
    return results_dict