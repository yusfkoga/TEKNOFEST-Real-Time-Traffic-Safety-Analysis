"""
TEKNOFEST 2026 — Yardımcı Fonksiyonlar.
Crop oluşturma, FPS sayacı.
Pipeline mantığından bağımsız, saf yardımcı araçlar.
"""

import time
from typing import List, Tuple, Optional

import numpy as np


def crop_with_padding(
    frame: np.ndarray,
    bbox: List[int],
    padding: float = 0.05,
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Bounding box'tan %padding kadar genişletilmiş crop alır.

    Args:
        frame: Tam kare (BGR)
        bbox: [x1, y1, x2, y2]
        padding: Her yöne uygulanacak genişletme oranı (0.05 = %5)

    Returns:
        (crop_image, x_offset, y_offset)
        crop_image None ise geçersiz crop demektir.
        Offset'ler crop-içi koordinatları frame koordinatlarına
        dönüştürmek için kullanılır: frame_x = crop_x + x_offset
    """
    if frame is None or frame.size == 0:
        return None, 0, 0

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return None, 0, 0

    # Padding hesapla — bbox boyutunun yüzdesi olarak
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    # Genişletilmiş sınırlar — frame dışına taşmayı önle
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    # .copy() — thread pool'a gönderilen crop orijinal frame'den bağımsız olmalı
    # Aksi hâlde YOLO/PaddleOCR inference sırasında frame GC'ye düşebilir
    crop = frame[cy1:cy2, cx1:cx2].copy()

    if crop.size == 0:
        return None, 0, 0

    return crop, cx1, cy1


class FPSCounter:
    """
    Kayan pencere (sliding window) tabanlı FPS sayacı.
    Anlık dalgalanmaları yumuşatır, gerçekçi ortalama verir.
    """

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Ortalama alınacak frame sayısı.
                         30 ≈ 1 saniyelik pencere @ 30fps.
        """
        from collections import deque as _deque
        self._timestamps: _deque = _deque(maxlen=window_size)
        self._window_size = window_size
        self._fps = 0.0

    def tick(self) -> None:
        """Yeni frame kaydeder ve FPS'i günceller."""
        now = time.perf_counter()
        self._timestamps.append(now)

        # En az 2 nokta gerekli — 1 aralık
        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self._fps = (len(self._timestamps) - 1) / elapsed

    @property
    def fps(self) -> float:
        """Güncel FPS değeri."""
        return self._fps
