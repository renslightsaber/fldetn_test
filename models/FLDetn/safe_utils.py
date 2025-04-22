import torch
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

# 🔒 1. Safe Global 등록 함수
def setup_safe_globals():
    try:
        from ultralytics.nn.tasks import DetectionModel
        from models.FLDetn.ultralytics.nn.modules import IEPR, ECM, CSNeck3in
    except ImportError as e:
        print(f"❌ 커스텀 모듈 import 실패: {e}")
        return

    classes = [DetectionModel, IEPR, ECM, CSNeck3in]
    torch.serialization.add_safe_globals(classes)
    print("✅ Safe globals 등록 완료.")


# ✅ 1.5. torch.load monkey patch
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    setup_safe_globals()
    kwargs["weights_only"] = False  # 🔥 핵심!
    return _original_torch_load(*args, **kwargs)

def patch_torch_load():
    torch.load = patched_torch_load
    print("✅ torch.load monkey patch 완료")


# 🧩 2. YOLO 객체 생성 시 safe_globals 등록하고 cfg 제거
def YOLO_safe(model_path, *args, **kwargs):
    patch_torch_load()  # ✅ 여기서 monkey patch 적용
    setup_safe_globals()
    kwargs.pop("cfg", None)
    kwargs.pop("model", None)
    return YOLO(model_path, *args, **kwargs)


# 🛡️ 3. YOLO.train 함수 monkey patch
YOLO_CLASS = YOLO

if not hasattr(YOLO_CLASS, "__original_train__"):
    YOLO_CLASS.__original_train__ = YOLO_CLASS.train

def patched_yolo_train(self, *args, **kwargs):
    print("🛡️ YOLO.train() 패치됨 — cfg 제거 & safe_globals 등록")
    setup_safe_globals()
    kwargs.pop("cfg", None)
    kwargs.pop("model", None)
    return YOLO_CLASS.__original_train__(self, *args, **kwargs)

YOLO_CLASS.train = patched_yolo_train
print("📌 패치 완료: YOLO.train")


# 💼 4. SafeTrainer 클래스 정의
class SafeTrainer(BaseTrainer):
    def __init__(self, overrides=None, _callbacks=None):
        print("🛠️ SafeTrainer 초기화")
        setup_safe_globals()
        if overrides is None:
            overrides = {}
        overrides.pop("cfg", None)
        overrides.pop("model", None)
        overrides["cfg"] = None
        overrides["pretrained"] = None
        super().__init__(overrides=overrides, _callbacks=_callbacks)

    def train(self):
        print("🛡️ SafeTrainer.train(): safe_globals 재등록")
        setup_safe_globals()
        return super().train()
