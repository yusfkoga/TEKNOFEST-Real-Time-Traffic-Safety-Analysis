"""
TEKNOFEST 2026 — YOLO Model Wrapper'ları.

Tier-1 : VehicleDetector    — yolo11s, tam kare, araç tespiti
Tier-2a: CabinAnalyzer      — yolo11s + SAHI, crop, araç içi nesne tespiti
Tier-2b: PlateDetectorYOLO  — yolo11s fine-tuned + SAHI, crop, plaka tespiti
         PlateDetectorHF    — HuggingFace fallback + SAHI

Model'ler __init__'de bir kez yüklenir.
Her frame'de yeniden yükleme KESİNLİKLE YASAKTIR.
"""

import logging
import math
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    COCO_LABELS,
    FACE_MESH_EAR_THRESHOLD,
    FACE_MESH_MAR_THRESHOLD,
    FACE_MESH_MIN_DETECTION_CONF,
    PLATE_DET_ASPECT_MAX,
    PLATE_DET_ASPECT_MIN,
    PLATE_HF_FILENAME,
    PLATE_HF_REPO,
    PLATE_MAX_WIDTH_RATIO,
    PLATE_MIN_WIDTH_RATIO,
    POSE_MIN_VISIBILITY,
    CABIN_MIN_CROP_PX,
    POSE_PHONE_THRESHOLD,
    SAHI_CONF,
    SAHI_OVERLAP_RATIO,
    SAHI_POSTPROCESS_MATCH_THRESHOLD,
    SAHI_POSTPROCESS_TYPE,
    SAHI_SLICE_HEIGHT,
    SAHI_SLICE_WIDTH,
    TENSORRT_ENABLED,
    TENSORRT_PRECISION,
    TIER1_CLASSES,
    TIER1_CONF,
    TIER1_IMGSZ,
    TIER1_MODEL_PATH,
    TIER2A_CLASSES,
    TIER2A_CONF,
    TIER2A_IMGSZ,
    TIER2A_MODEL_PATH,
    TIER2B_CONF,
    TIER2B_IMGSZ,
    TIER2B_MODEL_PATH,
)

logger = logging.getLogger(__name__)


def _load_yolo_model(model_path: str, imgsz: int = 640) -> YOLO:
    """
    YOLO model yükleyici — TensorRT native optimizasyon.

    Öncelik sırası:
      1. .engine varsa → direkt yükle          (~2-4ms, en hızlı)
      2. .pt → TensorRT export → .engine       (ilk seferde ~30-120sn, sonra anlık)
      3. .pt + half=True fallback               (~10-15ms, son çare)

    TensorRT engine boyut-sabit → dosya adında imgsz kodlanır:
      yolo11s_640.engine  (Tier-1, tam kare)
      yolo11s_320.engine  (Tier-2a, kabin crop)
    Aynı model farklı boyutlarda kullanıldığında çakışma olmaz.

    Export tek seferlik: GPU'ya özel compile edilir, sonraki çalıştırmalarda anlık yüklenir.
    """
    pt_path = Path(model_path)
    half = (TENSORRT_PRECISION == "fp16")
    int8 = (TENSORRT_PRECISION == "int8")

    # TensorRT engine: boyut-sabit → dosya adına imgsz kodla (çakışma önlemi)
    engine_path = pt_path.with_name(f"{pt_path.stem}_{imgsz}.engine")
    # Eski format da kontrol et (geriye uyumluluk)
    engine_path_legacy = pt_path.with_suffix(".engine")

    # ── 1. Önceden compile edilmiş engine var mı? ──
    for ep in (engine_path, engine_path_legacy):
        if ep.exists():
            logger.info(f"TensorRT engine bulundu: {ep}")
            return YOLO(str(ep))

    # ── 2. .pt yükle ──
    model = YOLO(model_path)

    if not TENSORRT_ENABLED:
        return model

    import torch
    if not torch.cuda.is_available():
        logger.warning("CUDA yok — TensorRT export atlandı, .pt ile devam")
        return model

    # ── 3. TensorRT native export ──
    # Tüm model tek parça olarak GPU'ya özel compile edilir.
    # Katman füzyonu + kernel autotuning + FP16/INT8 dönüşümü.
    try:
        logger.info(
            f"TensorRT export baslatiliyor: {model_path} → "
            f"{TENSORRT_PRECISION} (imgsz={imgsz})"
        )
        exported_path = model.export(
            format="engine",
            half=half,
            int8=int8,
            imgsz=imgsz,
            device=0,
        )
        # Ultralytics {stem}.engine olarak kaydeder — imgsz-kodlu isme taşı
        exported = Path(exported_path)
        if exported.exists() and exported != engine_path:
            target = exported.with_name(f"{pt_path.stem}_{imgsz}.engine")
            exported.rename(target)
            exported_path = str(target)
            logger.info(f"Engine rename: {exported.name} → {target.name}")
        logger.info(f"TensorRT export tamamlandi: {exported_path}")
        return YOLO(exported_path)
    except Exception as e:
        logger.warning(f"TensorRT export basarisiz: {e} — .pt ile devam")

    # ── 4. Fallback: PyTorch .pt ──
    logger.info(f"Fallback: PyTorch .pt ile devam ({model_path})")
    return model


def _create_sahi_wrapper(existing_model: YOLO, model_path: str, conf: float,
                         device: str = "cuda:0"):
    """
    Mevcut YOLO modelini SAHI DetectionModel wrapper'ına sarar.
    Duplikat model yüklemesini önler → GPU bellek tasarrufu.

    SAHI'nin AutoDetectionModel.from_pretrained() normalde AYNI modeli
    ikinci kez GPU'ya yükler. Bu fonksiyon load_model()'i override ederek
    zaten yüklü olan modeli paylaştırır.
    """
    from sahi.models.ultralytics import UltralyticsDetectionModel

    class _ReusableModel(UltralyticsDetectionModel):
        """load_model override — duplikat GPU yüklemesini engeller."""
        def load_model(self):
            pass  # Model dışarıdan set edilecek

    wrapper = _ReusableModel(
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )
    wrapper.model = existing_model
    # SAHI'nin sınıf eşleme tablosunu mevcut modelden al
    if hasattr(existing_model, "names") and existing_model.names:
        wrapper.category_mapping = {
            str(k): v for k, v in existing_model.names.items()
        }
    return wrapper


class VehicleDetector:
    """
    Tier-1: Araç Tespiti + BoTSORT Tracking.

    Model  : yolo11s.pt (hız öncelikli)
    Girdi  : TAM KARE (640px)
    Çalışma: Her frame
    Classes: [2, 3, 5, 7] — car, motorcycle, bus, truck
    conf   : 0.40

    Tracking:
      model.track(persist=True, tracker="botsort.yaml")
      Kalman filtre + ReID + IoU assignment.
      Her tespit track_id ile birlikte döner.
    """

    def __init__(self):
        logger.info(f"Tier-1 model yukleniyor: {TIER1_MODEL_PATH}")
        self.model = _load_yolo_model(TIER1_MODEL_PATH, imgsz=TIER1_IMGSZ)

        # Warmup — ilk inference yavaş olur, dummy frame ile ısıt
        # Bu sayede gerçek ilk frame'de daha stabil FPS elde edilir
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, imgsz=TIER1_IMGSZ, verbose=False)
        logger.info("Tier-1 model hazir (warmup tamamlandi)")

    def track(self, frame: np.ndarray) -> List[Dict]:
        """
        Frame'deki araçları tespit eder VE BoTSORT ile takip eder.
        Her tespit track_id ile birlikte döner.

        Args:
            frame: BGR formatında tam kare

        Returns:
            [{"bbox": [x1,y1,x2,y2], "track_id": str,
              "class_id": int, "confidence": float, "label": str}, ...]
        """
        results = self.model.track(
            frame,
            imgsz=TIER1_IMGSZ,
            conf=TIER1_CONF,
            classes=TIER1_CLASSES,
            persist=True,
            tracker="botsort.yaml",
            half=True,
            verbose=False,
        )

        tracked = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # BoTSORT ID'leri boxes.id'de bulunur
            if boxes.id is None:
                continue

            # Batch GPU→CPU: tek .cpu() çağrısı, N ayrı sync yerine
            xyxy_np = boxes.xyxy.cpu().numpy().astype(int)
            cls_np = boxes.cls.cpu().numpy().astype(int)
            conf_np = boxes.conf.cpu().numpy()
            id_np = boxes.id.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                cls_id = int(cls_np[i])
                tracked.append(
                    {
                        "bbox": xyxy_np[i].tolist(),
                        "track_id": str(int(id_np[i])),
                        "class_id": cls_id,
                        "confidence": float(conf_np[i]),
                        "label": COCO_LABELS.get(cls_id, f"class_{cls_id}"),
                    }
                )

        return tracked


class CabinAnalyzer:
    """
    Tier-2a: Kabin Kişi Tespiti.

    Model  : yolo11s.pt (TensorRT engine)
    Girdi  : ARAÇ CROP'U (üst %60)
    Çalışma: Her CABIN_EVERY_N frame
    Classes: [0] — person
    conf   : 0.40
    imgsz  : 320

    Amaç: Kabinde sürücü var mı? → Evet ise MediaPipe Pose/FaceMesh'e gönder.
    Telefon/sigara tespiti MediaPipe davranış analizine devredildi.
    Minimum crop boyutu: CABIN_MIN_CROP_PX (bulanık büyütme koruması).
    """

    def __init__(self):
        self.ready = False
        try:
            logger.info(f"Tier-2a model yukleniyor: {TIER2A_MODEL_PATH}")
            self.model = _load_yolo_model(TIER2A_MODEL_PATH, imgsz=TIER2A_IMGSZ)
            
            # Warmup
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model(dummy, imgsz=TIER2A_IMGSZ, verbose=False)
            self.ready = True
            logger.info("Tier-2a model hazir (warmup tamamlandi)")

        except Exception as e:
            logger.warning(f"[CabinAnalyzer] Fallback modu devrede. Neden: {e}")

    def detect(self, crop: np.ndarray) -> List[Dict]:
        """
        Araç crop'undaki kabin nesnelerini tespit eder.

        Args:
            crop: BGR formatında araç kesimi

        Returns:
            [{"bbox": [x1,y1,x2,y2], "class_id": int,
              "confidence": float, "label": str, "mode": ...}, ...]
        """
        if not self.ready:
            return []

        if crop is None or crop.size == 0:
            return []

        h, w = crop.shape[:2]
        if h < CABIN_MIN_CROP_PX or w < CABIN_MIN_CROP_PX:
            return []

        results = self.model(
            crop,
            imgsz=TIER2A_IMGSZ,
            conf=TIER2A_CONF,
            classes=TIER2A_CLASSES,
            half=True,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            # Batch GPU→CPU: tek .cpu() çağrısı
            xyxy_np = boxes.xyxy.cpu().numpy().astype(int)
            cls_np = boxes.cls.cpu().numpy().astype(int)
            conf_np = boxes.conf.cpu().numpy()

            for i in range(len(boxes)):
                cls_id = int(cls_np[i])
                detections.append(
                    {
                        "bbox": xyxy_np[i].tolist(),
                        "class_id": cls_id,
                        "confidence": float(conf_np[i]),
                        "label": COCO_LABELS.get(cls_id, f"class_{cls_id}"),
                    }
                )

        return detections

    def detect_with_sahi(self, crop: np.ndarray) -> List[Dict]:
        """SAHI Tier-2a'da kullanılmaz — person büyük nesne, SAHI gereksiz. Direkt detect()."""
        return self.detect(crop)


class PlateDetectorYOLO:
    """
    Tier-2b: Plaka Tespiti (YOLO Fine-Tuned).

    Model  : models/plate_yolo11s_turkiye.pt
    Girdi  : ARAÇ CROP'U
    conf   : 0.15
    imgsz  : 640

    Sadece fine-tuned lokal model varsa kullanılır.
    """

    def __init__(self):
        logger.info(f"Tier-2b YOLO model yukleniyor: {TIER2B_MODEL_PATH}")
        self.model = _load_yolo_model(TIER2B_MODEL_PATH, imgsz=TIER2B_IMGSZ)
        self.sahi_model = None

        # Warmup
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model(dummy, imgsz=TIER2B_IMGSZ, verbose=False)
        logger.info("Tier-2b YOLO model hazir (warmup tamamlandi)")

        # SAHI wrapper — mevcut modeli paylaşır, duplikat yükleme YOK
        try:
            self.sahi_model = _create_sahi_wrapper(
                self.model, TIER2B_MODEL_PATH, TIER2B_CONF,
            )
            logger.info("SAHI model hazir (PlateDetectorYOLO) — mevcut model paylasild")
        except Exception as e:
            logger.warning(f"SAHI plate yuklenemedi: {e}")

    def detect(
        self, crop: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        """Araç crop'unda plaka tespit eder."""
        if crop is None or crop.size == 0:
            return None, None
        results = self.model(
            crop, imgsz=TIER2B_IMGSZ, conf=TIER2B_CONF, half=True, verbose=False,
        )
        return _pick_best_plate(results, crop)

    def detect_with_sahi(
        self, crop: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        """SAHI ile plaka tespiti — QoD tetiklendiğinde çağrılır."""
        if self.sahi_model is None:
            return self.detect(crop)
        return _sahi_plate_predict(self.sahi_model, crop)


class PlateDetectorHF:
    """
    Tier-2b Fallback: HuggingFace Plaka Modeli.

    Model  : morsetechlab/yolov11-license-plate-detection (l varyantı)
    Girdi  : ARAÇ CROP'U
    conf   : 0.25
    imgsz  : 640

    YOLO11 tabanlı — ultralytics ile direkt uyumlu.
    Fine-tuned lokal model yoksa otomatik devreye girer.
    Model ilk çalıştırmada huggingface_hub ile indirilir ve cache'lenir.
    """

    def __init__(self):
        from huggingface_hub import hf_hub_download

        logger.info(
            f"HuggingFace plaka modeli indiriliyor: "
            f"{PLATE_HF_REPO}/{PLATE_HF_FILENAME}"
        )
        model_path = hf_hub_download(
            repo_id=PLATE_HF_REPO,
            filename=PLATE_HF_FILENAME,
        )
        logger.info(f"Model indirildi: {model_path}")
        self.model = YOLO(model_path)
        self.sahi_model = None

        # Warmup
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model(dummy, imgsz=TIER2B_IMGSZ, verbose=False)
        logger.info("HuggingFace plaka modeli hazir (warmup tamamlandi)")

        # SAHI wrapper — mevcut modeli paylaşır, duplikat yükleme YOK
        try:
            self.sahi_model = _create_sahi_wrapper(
                self.model, model_path, TIER2B_CONF,
            )
            logger.info("SAHI model hazir (PlateDetectorHF) — mevcut model paylasild")
        except Exception as e:
            logger.warning(f"SAHI plate HF yuklenemedi: {e}")

    def detect(
        self, crop: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        """Araç crop'unda plaka tespit eder (PlateDetectorYOLO ile aynı interface)."""
        if crop is None or crop.size == 0:
            return None, None
        results = self.model(
            crop, imgsz=TIER2B_IMGSZ, conf=TIER2B_CONF, half=True, verbose=False,
        )
        return _pick_best_plate(results, crop)

    def detect_with_sahi(
        self, crop: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        """SAHI ile plaka tespiti — QoD tetiklendiğinde çağrılır."""
        if self.sahi_model is None:
            return self.detect(crop)
        return _sahi_plate_predict(self.sahi_model, crop)


def _score_plate_candidates(
    candidates: List[Tuple[float, List[int]]],
    crop: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
    """
    Plaka adaylarını skorlar, en iyisini seçer, pad'li crop döndürür.
    _pick_best_plate ve _sahi_plate_predict ortak kullanır.

    Args:
        candidates: [(confidence, [x1,y1,x2,y2]), ...]
        crop: Orijinal araç crop'u (BGR)

    Returns:
        (plate_bgr_padded, [x1,y1,x2,y2]) veya (None, None)
    """
    crop_h, crop_w = crop.shape[:2]
    best_score = -999.0
    best_bbox = None

    for conf, bbox in candidates:
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        if bw <= 0 or bh <= 0:
            continue

        aspect = bw / bh
        if aspect < PLATE_DET_ASPECT_MIN or aspect > PLATE_DET_ASPECT_MAX:
            continue

        width_ratio = bw / crop_w
        if width_ratio < PLATE_MIN_WIDTH_RATIO or width_ratio > PLATE_MAX_WIDTH_RATIO:
            continue

        cx = bbox[0] + bw / 2
        cy = bbox[1] + bh / 2
        x_ratio = cx / crop_w
        y_ratio = cy / crop_h

        if x_ratio < 0.20 or x_ratio > 0.80:
            continue
        if y_ratio < 0.30:
            continue

        aspect_diff = abs(aspect - 4.7)
        center_dist = abs(x_ratio - 0.5) * 2
        score = conf - (aspect_diff * 0.4) - (center_dist * 0.3)

        if score > best_score:
            best_score = score
            best_bbox = bbox

    if best_bbox is None:
        return None, None

    bw = best_bbox[2] - best_bbox[0]
    bh = best_bbox[3] - best_bbox[1]
    pad_x = int(bw * 0.15)
    pad_y = int(bh * 0.15)

    cx1 = max(0, best_bbox[0] - pad_x)
    cy1 = max(0, best_bbox[1] - pad_y)
    cx2 = min(crop_w, best_bbox[2] + pad_x)
    cy2 = min(crop_h, best_bbox[3] + pad_y)

    if cx2 <= cx1 or cy2 <= cy1:
        return None, None

    plate_crop = crop[cy1:cy2, cx1:cx2].copy()
    return plate_crop, best_bbox


def _pick_best_plate(
    results, crop: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
    """
    YOLO çıktısından en iyi plaka adayını seçer.
    PlateDetectorYOLO ve PlateDetectorHF ortak kullanır.
    """
    candidates = []
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy_np = boxes.xyxy.cpu().numpy().astype(int)
        conf_np = boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            candidates.append((float(conf_np[i]), xyxy_np[i].tolist()))

    return _score_plate_candidates(candidates, crop)


def _sahi_plate_predict(
    sahi_model, crop: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
    """
    SAHI ile plaka tespiti yapıp aynı (plate_crop, bbox) formatında döndürür.
    """
    if crop is None or crop.size == 0:
        return None, None

    try:
        from sahi.predict import get_sliced_prediction

        result = get_sliced_prediction(
            image=crop,
            detection_model=sahi_model,
            slice_height=SAHI_SLICE_HEIGHT,
            slice_width=SAHI_SLICE_WIDTH,
            overlap_height_ratio=SAHI_OVERLAP_RATIO,
            overlap_width_ratio=SAHI_OVERLAP_RATIO,
            postprocess_type=SAHI_POSTPROCESS_TYPE,
            postprocess_match_threshold=SAHI_POSTPROCESS_MATCH_THRESHOLD,
            verbose=0,
        )

        candidates = [
            (pred.score.value, [int(v) for v in pred.bbox.to_xyxy()])
            for pred in result.object_prediction_list
        ]

        return _score_plate_candidates(candidates, crop)

    except Exception as e:
        logger.warning(f"SAHI plate inference hatasi: {e}")
        return None, None


def create_plate_detector():
    """
    Plaka tespit modeli fabrikası.

    Öncelik sırası:
        1. Fine-tuned YOLO model (models/plate_yolo11s_turkiye.pt) — en doğru
        2. HuggingFace model (morsetechlab) — YOLO11 tabanlı, uyumlu
        3. Klasik CV fallback — son çare

    Bu sıralama önemlidir:
    - Fine-tuned model Türk plakalarına özeldir → en iyi sonuç
    - HuggingFace model genel plaka tespiti yapar → orta sonuç
    - CV fallback Sobel+contour → en zayıf ama model gerektirmez
    """
    model_path = Path(TIER2B_MODEL_PATH)

    if model_path.exists():
        logger.info(
            f"Fine-tuned plaka modeli bulundu: {model_path} "
            "-> YOLO plate detector aktif"
        )
        return PlateDetectorYOLO()
    else:
        logger.info(
            f"Fine-tuned plaka modeli bulunamadi: {model_path} "
            f"-> HuggingFace fallback aktif: {PLATE_HF_REPO}/{PLATE_HF_FILENAME}"
        )
        try:
            return PlateDetectorHF()
        except Exception as e:
            logger.warning(
                f"HuggingFace model yuklenemedi: {e} "
                "-> Klasik CV fallback aktif"
            )
            from plate_cv import PlateDetectorCV
            return PlateDetectorCV()


# ══════════════════════════════════════════════════════
# Plaka Oriented Bounding Box (OBB) Refinement
# ══════════════════════════════════════════════════════


def refine_plate_contour(
    crop: np.ndarray, bbox: List[int]
) -> Optional[np.ndarray]:
    """
    YOLO axis-aligned bbox → OpenCV oriented bounding box.

    Plaka açılı durduğunda (perspektif, kamera açısı) axis-aligned
    dikdörtgen plakayı sıkı saramaz. Bu fonksiyon kenar tespiti ile
    gerçek plaka konturunu bulup döndürülmüş 4 köşe noktası döner.

    Args:
        crop: Araç crop'u (BGR)
        bbox: YOLO axis-aligned [x1, y1, x2, y2] (crop-yerel)

    Returns:
        np.ndarray shape (4,2) — 4 köşe noktası (crop-yerel koordinat)
        veya None (kontur bulunamazsa)
    """
    x1, y1, x2, y2 = bbox
    roi = crop[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Kontrast artırma (karanlık sahneler)

    edges = cv2.Canny(gray, 40, 120)

    # Geniş dilation: plaka karakterlerini tek büyük bloba birleştir
    # Yatay kernel: karakterler yatayda yanyana → yatay morfoloji daha etkili
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    merged = cv2.dilate(edges, h_kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Tüm kontürlerin birleşik convex hull'u → plaka sınırını kapsar
    all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    if len(all_pts) < 4:
        return None

    hull = cv2.convexHull(all_pts)

    # Minimum alan döndürülmüş dikdörtgen
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # ROI-yerel → crop-yerel koordinata çevir
    box[:, 0] += x1
    box[:, 1] += y1

    return box


# ══════════════════════════════════════════════════════
# Pose Tabanlı Davranış Analizi (MediaPipe)
# ══════════════════════════════════════════════════════

_pose_local = threading.local()
_face_mesh_local = threading.local()


class PoseAnalyzer:
    """
    MediaPipe Tasks API tabanlı telefon ve sigara kullanımı tespiti.

    YOLO "cell phone" ve sigara sınıfları cam arkasından başarısız olduğu için
    bilek ↔ kulak (telefon) ve bilek ↔ ağız (sigara) L2 mesafesine
    dayalı davranış tespiti yapılır.
    Referans: e-candeloro/Driver-State-Detection (aynı L2 pattern).

    mediapipe 0.10.33+ → legacy solutions API kaldırıldı,
    Tasks API (PoseLandmarker) kullanılır.

    Thread-safety: PoseLandmarker instance'ı thread-safe değildir.
    Her thread kendi instance'ını _pose_local üzerinden alır.
    """

    # Landmark indeksleri (COCO-33 topolojisi)
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    def __init__(self):
        self.ready = False
        self._model_path = None
        try:
            import mediapipe as mp  # noqa
            from mediapipe.tasks.python import vision as _  # noqa — API check

            model_path = Path(__file__).parent / "models" / "pose_landmarker_lite.task"
            if not model_path.exists():
                logger.warning(
                    f"Pose model bulunamadi: {model_path} — PoseAnalyzer devre disi"
                )
                return
            self._model_path = str(model_path)
            # Ana thread'de bir test instance oluştur — model'in yüklendiğini doğrula
            self._create_landmarker().close()
            self.ready = True
            logger.info(f"PoseAnalyzer hazir (MediaPipe Tasks API, model={model_path.name})")
        except ImportError:
            logger.warning("MediaPipe yuklenemedi — PoseAnalyzer devre disi")
        except Exception as e:
            logger.warning(f"PoseAnalyzer init hatasi: {e}")

    def _create_landmarker(self):
        """Yeni PoseLandmarker instance oluşturur."""
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def _get_landmarker(self):
        """Thread-local PoseLandmarker instance döndürür."""
        if not hasattr(_pose_local, "landmarker"):
            _pose_local.landmarker = self._create_landmarker()
        return _pose_local.landmarker

    def analyze(self, crop_bgr: np.ndarray) -> dict:
        """
        Araç crop'unda telefon kullanımını tespit eder.

        Args:
            crop_bgr: BGR formatında araç crop'u (tam araç — MediaPipe kişiyi kendi bulur)

        Returns:
            {"phone_call": bool, "score": float, "side": str,
             "smoking": bool, "smoking_score": float, "smoking_side": str}
        """
        result = {
            "phone_call": False, "score": 0.0, "side": "",
            "smoking": False, "smoking_score": 0.0, "smoking_side": "",
        }

        if not self.ready or crop_bgr is None or crop_bgr.size == 0:
            return result

        try:
            import mediapipe as mp

            landmarker = self._get_landmarker()

            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = landmarker.detect(mp_image)

            if not detection.pose_landmarks:
                logger.debug(
                    f"POSE_NO_LANDMARKS | crop={crop_bgr.shape[1]}x{crop_bgr.shape[0]}"
                )
                return result

            lm = detection.pose_landmarks[0]  # İlk (ve tek) kişi

            # ── Ağız orta noktasını hesapla ──
            ml = lm[self.MOUTH_LEFT]
            mr = lm[self.MOUTH_RIGHT]
            mouth_vis = min(ml.visibility, mr.visibility)
            mouth_x = (ml.x + mr.x) / 2.0
            mouth_y = (ml.y + mr.y) / 2.0

            best_ear_dist = float("inf")
            best_ear_side = ""
            best_mouth_dist = float("inf")
            best_mouth_side = ""

            for wrist_id, ear_id, side in [
                (self.RIGHT_WRIST, self.RIGHT_EAR, "right"),
                (self.LEFT_WRIST, self.LEFT_EAR, "left"),
            ]:
                w = lm[wrist_id]
                e = lm[ear_id]

                w_vis = w.visibility
                e_vis = e.visibility

                if w_vis < POSE_MIN_VISIBILITY:
                    continue

                # ── Telefon: bilek ↔ kulak ──
                if e_vis >= POSE_MIN_VISIBILITY:
                    dist_ear = math.hypot(w.x - e.x, w.y - e.y)
                    if dist_ear < best_ear_dist:
                        best_ear_dist = dist_ear
                        best_ear_side = side

                # ── Sigara: bilek ↔ ağız ──
                if mouth_vis >= POSE_MIN_VISIBILITY:
                    dist_mouth = math.hypot(w.x - mouth_x, w.y - mouth_y)
                    if dist_mouth < best_mouth_dist:
                        best_mouth_dist = dist_mouth
                        best_mouth_side = side

            # ── Mesafe karşılaştırmalı karar ──
            # Bilek hem kulağa hem ağza yakın olabilir.
            # Hangisine DAHA yakınsa o davranış seçilir (mutual exclusion).
            smoking_threshold = POSE_PHONE_THRESHOLD * 1.2

            phone_triggered = (
                best_ear_dist < float("inf")
                and best_ear_dist < POSE_PHONE_THRESHOLD
            )
            smoking_triggered = (
                best_mouth_dist < float("inf")
                and best_mouth_dist < smoking_threshold
            )

            if phone_triggered and smoking_triggered:
                # İkisi de eşik altında → daha yakın olan kazanır
                if best_mouth_dist < best_ear_dist:
                    phone_triggered = False   # Ağza daha yakın → sigara
                else:
                    smoking_triggered = False  # Kulağa daha yakın → telefon

            # ── Telefon sonuçları ──
            if phone_triggered:
                score = max(0.0, 1.0 - best_ear_dist / POSE_PHONE_THRESHOLD)
                result["phone_call"] = True
                result["score"] = round(score, 2)
                result["side"] = best_ear_side

            # ── Sigara sonuçları ──
            if smoking_triggered:
                smoking_score = max(0.0, 1.0 - best_mouth_dist / smoking_threshold)
                result["smoking"] = True
                result["smoking_score"] = round(smoking_score, 2)
                result["smoking_side"] = best_mouth_side

            return result

        except Exception as e:
            logger.debug(f"Pose analiz hatasi: {e}")
            return result


class FaceMeshAnalyzer:
    """
    MediaPipe Face Mesh tabanlı sürücü yorgunluk/dikkat analizi.

    EAR (Eye Aspect Ratio):
        Göz açıklığı oranı. 6 landmark ile hesaplanır.
        EAR < eşik → göz kapalı sayılır.

    PERCLOS (Percentage of Eye Closure):
        Son N frame'de göz kapalılık yüzdesi.
        PERCLOS > %40 → uyuklama (ISO 17938 standardı).

    MAR (Mouth Aspect Ratio):
        Ağız açıklığı oranı. MAR > eşik → esneme.

    Thread-safety: FaceLandmarker thread-safe değildir.
    Her thread kendi instance'ını _face_mesh_local üzerinden alır.
    """

    # MediaPipe Face Mesh landmark indeksleri (468 nokta topolojisi)
    # Sol göz
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    # Sağ göz
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    # Ağız (iç dudak)
    MOUTH = [13, 14, 78, 308, 81, 311]  # üst, alt, sol, sağ, sol-alt, sağ-alt

    def __init__(self):
        self.ready = False
        self._model_path = None
        try:
            import mediapipe as mp  # noqa
            from mediapipe.tasks.python import vision as _  # noqa — API check

            model_path = Path(__file__).parent / "models" / "face_landmarker.task"
            if not model_path.exists():
                logger.warning(
                    f"Face Mesh model bulunamadi: {model_path} — FaceMeshAnalyzer devre disi"
                )
                return
            self._model_path = str(model_path)
            # Ana thread'de test instance
            self._create_landmarker().close()
            self.ready = True
            logger.info(f"FaceMeshAnalyzer hazir (MediaPipe Tasks API, model={model_path.name})")
        except ImportError:
            logger.warning("MediaPipe yuklenemedi — FaceMeshAnalyzer devre disi")
        except Exception as e:
            logger.warning(f"FaceMeshAnalyzer init hatasi: {e}")

    def _create_landmarker(self):
        """Yeni FaceLandmarker instance oluşturur."""
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python import BaseOptions

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=FACE_MESH_MIN_DETECTION_CONF,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.3,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _get_landmarker(self):
        """Thread-local FaceLandmarker instance döndürür."""
        if not hasattr(_face_mesh_local, "landmarker"):
            _face_mesh_local.landmarker = self._create_landmarker()
        return _face_mesh_local.landmarker

    @staticmethod
    def _ear(landmarks, eye_indices) -> float:
        """
        Eye Aspect Ratio hesaplar.
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        p1,p4 = yatay köşeler, p2-p3,p5-p6 = dikey çiftler
        """
        p = [landmarks[i] for i in eye_indices]
        # Dikey mesafeler
        v1 = math.hypot(p[1].x - p[5].x, p[1].y - p[5].y)
        v2 = math.hypot(p[2].x - p[4].x, p[2].y - p[4].y)
        # Yatay mesafe
        h = math.hypot(p[0].x - p[3].x, p[0].y - p[3].y)
        if h < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    @staticmethod
    def _mar(landmarks) -> float:
        """
        Mouth Aspect Ratio hesaplar.
        MAR = (|üst-alt| + |sol_alt-sağ_alt|) / (2 * |sol-sağ|)
        """
        top = landmarks[13]   # üst dudak iç
        bot = landmarks[14]   # alt dudak iç
        left = landmarks[78]  # sol köşe
        right = landmarks[308]  # sağ köşe
        l_bot = landmarks[81]
        r_bot = landmarks[311]

        v1 = math.hypot(top.x - bot.x, top.y - bot.y)
        v2 = math.hypot(l_bot.x - r_bot.x, l_bot.y - r_bot.y)
        h = math.hypot(left.x - right.x, left.y - right.y)
        if h < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * h)

    def analyze(self, crop_bgr: np.ndarray) -> dict:
        """
        Kabin crop'unda yüz analizi yapar.

        Args:
            crop_bgr: BGR formatında kabin crop'u

        Returns:
            {"ear_left": float, "ear_right": float, "ear_avg": float,
             "eye_closed": bool, "mar": float, "yawn": bool}
        """
        result = {
            "ear_left": 0.0, "ear_right": 0.0, "ear_avg": 0.0,
            "eye_closed": False, "mar": 0.0, "yawn": False,
        }

        if not self.ready or crop_bgr is None or crop_bgr.size == 0:
            return result

        try:
            import mediapipe as mp

            landmarker = self._get_landmarker()
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = landmarker.detect(mp_image)

            if not detection.face_landmarks:
                return result

            lm = detection.face_landmarks[0]

            # EAR
            ear_l = self._ear(lm, self.LEFT_EYE)
            ear_r = self._ear(lm, self.RIGHT_EYE)
            ear_avg = (ear_l + ear_r) / 2.0

            result["ear_left"] = round(ear_l, 3)
            result["ear_right"] = round(ear_r, 3)
            result["ear_avg"] = round(ear_avg, 3)
            result["eye_closed"] = ear_avg < FACE_MESH_EAR_THRESHOLD

            # MAR
            mar_val = self._mar(lm)
            result["mar"] = round(mar_val, 3)
            result["yawn"] = mar_val > FACE_MESH_MAR_THRESHOLD

            return result

        except Exception as e:
            logger.debug(f"FaceMesh analiz hatasi: {e}")
            return result

