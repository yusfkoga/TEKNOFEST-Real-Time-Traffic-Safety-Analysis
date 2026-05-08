"""
TEKNOFEST 2026 — BoTSORT Tabanlı Araç Tracker.

Ultralytics built-in BoTSORT kullanılır.
model.track(persist=True, tracker="botsort.yaml") ile ID ataması
Kalman filtre + ReID + IoU assignment = güçlü takip.

TrackManager sınıfı BoTSORT çıktısını TrackState'e eşler.
Pipeline boyunca TrackState üzerinden bilgi paylaşılır.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import (
    BBOX_HISTORY_LEN,
    FACE_MESH_PERCLOS_WINDOW,
    MAX_AGE,
    MAX_TRACKS,
)

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    """
    Tek bir araç için takip durumu.
    Pipeline boyunca bu nesne üzerinden bilgi paylaşılır.
    Thread-safety notu: ana async loop sıralı erişim sağlar,
    sadece OCR worker ayrı thread'den plate_text yazar.
    """

    track_id: str                                  # BoTSORT atadığı ID (int→str)
    bbox_history: deque                            # Son N frame bbox'ıları [x1,y1,x2,y2]
    last_seen_frame: int                           # Son görüldüğü frame numarası

    # Araç tespiti metadata
    class_id: int = 2                              # COCO sınıf ID (varsayılan: car)
    label: str = "car"                             # İnsan-okunur etiket
    detection_conf: float = 0.0                    # Son tespit güven değeri

    # Plaka durumu
    plate_text: str = ""                           # Doğrulanmış plaka metni
    plate_confidence: float = 0.0                  # OCR güven değeri (0.0–1.0)
    plate_last_ocr_frame: int = -999               # Son OCR frame'i (rate limiting)
    plate_bbox_abs: Optional[List[int]] = None     # Plaka bbox mutlak koordinatları
    plate_corners_abs: Optional[np.ndarray] = None # Oriented bbox — 4 köşe noktası [[x,y],...]

    # Kabin analizi
    cabin_objects: List[Dict] = field(default_factory=list)
    # Her eleman: {"bbox": [x1,y1,x2,y2], "class_id": int,
    #              "confidence": float, "label": str}

    # Pose analizi (MediaPipe)
    phone_detected: bool = False                   # Bilek ↔ Kulak mesafesi eşiği altında
    phone_score: float = 0.0                       # 0-1: 1 = kulağa dayalı
    cigarette_detected: bool = False               # Bilek ↔ Ağız mesafesi eşiği altında
    cigarette_score: float = 0.0                   # 0-1: 1 = ağza dayalı

    # Face Mesh analizi
    ear_left: float = 0.0                          # Sol göz EAR (Eye Aspect Ratio)
    ear_right: float = 0.0                         # Sağ göz EAR
    perclos: float = 0.0                           # PERCLOS: son N frame'de göz kapalılık oranı
    mar: float = 0.0                               # MAR (Mouth Aspect Ratio) — esneme
    drowsy_detected: bool = False                  # PERCLOS eşiği aşıldı
    yawn_detected: bool = False                    # MAR eşiği aşıldı
    _eye_closed_history: deque = field(
        default_factory=lambda: deque(maxlen=FACE_MESH_PERCLOS_WINDOW)
    )  # PERCLOS penceresi — deque otomatik eviction

    # Kritiklik
    criticality_score: float = 0.0                 # Her frame güncellenir (0–100)

    # Pipeline state
    last_t2_frame: int = -999                      # Son Tier-2 analiz frame'i


class TrackManager:
    """
    BoTSORT çıktısını TrackState nesnelerine eşleyen yönetici.

    BoTSORT ID atamasını ultralytics yapar (Kalman + ReID + IoU).
    TrackManager sadece:
        1. Yeni ID → yeni TrackState oluştur
        2. Mevcut ID → TrackState güncelle (bbox, conf, class)
        3. MAX_AGE frame görünmeyen → sil
        4. MAX_TRACKS aşılırsa en eski track'ı evict et
    """

    def __init__(self):
        self._tracks: Dict[str, TrackState] = {}

    @property
    def active_count(self) -> int:
        """Aktif track sayısı."""
        return len(self._tracks)

    def get_active_tracks(self) -> List[TrackState]:
        """Tüm aktif track'ları döndürür."""
        return list(self._tracks.values())

    def get_track(self, track_id: str) -> Optional[TrackState]:
        """Belirli bir track'ı ID ile getirir."""
        return self._tracks.get(track_id)

    def update(self, tracked_detections: List[Dict], frame_idx: int) -> None:
        """
        BoTSORT çıktısını TrackState'lere eşler.

        Args:
            tracked_detections: [{"bbox": [x1,y1,x2,y2], "track_id": str,
                                  "class_id": int, "confidence": float,
                                  "label": str}, ...]
            frame_idx: Mevcut frame numarası
        """
        for det in tracked_detections:
            tid = det["track_id"]

            if tid in self._tracks:
                # Mevcut track — güncelle
                track = self._tracks[tid]
                track.bbox_history.append(det["bbox"])
                track.last_seen_frame = frame_idx
                track.class_id = det["class_id"]
                track.label = det["label"]
                track.detection_conf = det["confidence"]
            else:
                # Yeni track — oluştur
                track = TrackState(
                    track_id=tid,
                    bbox_history=deque([det["bbox"]], maxlen=BBOX_HISTORY_LEN),
                    last_seen_frame=frame_idx,
                    class_id=det.get("class_id", 2),
                    label=det.get("label", "car"),
                    detection_conf=det.get("confidence", 0.0),
                )
                self._tracks[tid] = track
                logger.debug(
                    f"Yeni track: {tid} | "
                    f"class={track.label} | conf={track.detection_conf:.2f}"
                )

        self._enforce_max_tracks()

    def cleanup(self, current_frame: int) -> int:
        """
        Ölü track'ları temizler.

        Args:
            current_frame: Mevcut frame numarası

        Returns:
            Silinen track sayısı
        """
        stale_ids = [
            tid
            for tid, track in self._tracks.items()
            if current_frame - track.last_seen_frame > MAX_AGE
        ]

        for tid in stale_ids:
            del self._tracks[tid]

        return len(stale_ids)

    def _enforce_max_tracks(self) -> None:
        """MAX_TRACKS aşılırsa en eski track'ları siler (last_seen_frame'e göre)."""
        while len(self._tracks) > MAX_TRACKS:
            oldest_tid = min(
                self._tracks,
                key=lambda tid: self._tracks[tid].last_seen_frame,
            )
            del self._tracks[oldest_tid]
            logger.debug(f"LRU eviction: track {oldest_tid}")
