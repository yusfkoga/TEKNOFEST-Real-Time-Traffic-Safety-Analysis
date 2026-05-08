"""
TEKNOFEST 2026 — Klasik CV Tabanlı Plaka Tespiti (Fallback).

Fine-tuned YOLO modeli (plate_yolo11s.pt) yoksa otomatik devreye girer.
Türk plakalarının fiziksel sabitlerini kullanır:
    Boyut  : 520×110mm → oran ~4.7:1
    Renk   : Beyaz zemin, siyah karakter
    Kenar  : Belirgin dikdörtgen

Pipeline:
    1. Bilateral filter (kenar koruyarak gürültü azalt)
    2. Sobel edge detection
    3. Kontur bul
    4. Aspect ratio filtrele (3.5 < oran < 5.5)
    5. Alan filtrele (min/max px²)
    6. Perspektif düzelt → OCR'a ilet

Aynı interface'i döndürür: (plate_bgr, bbox) veya (None, None)
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    PLATE_ASPECT_MAX,
    PLATE_ASPECT_MIN,
    PLATE_MAX_AREA,
    PLATE_MIN_AREA,
    PLATE_MAX_WIDTH_RATIO,
)
logger = logging.getLogger(__name__)


def _order_points(pts: np.ndarray) -> np.ndarray:
    """4 noktayı tutarlı sıraya koyar: [sol-üst, sağ-üst, sağ-alt, sol-alt]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


class PlateDetectorCV:
    """
    Klasik bilgisayarla görme tabanlı Türk plaka tespiti.

    Avantajlar:
        - Ek model eğitimi gerektirmez
        - CPU'da hızlı çalışır
        - Türk plakalarının geometrik özelliklerine özelleştirilmiş

    Dezavantajlar:
        - YOLO'ya göre daha fazla yanlış pozitif
        - Aşırı perspektif bozulmalarında zayıf
        - Gece/düşük ışıkta düşük performans

    Fine-tuned YOLO modeli hazır olduğunda devre dışı kalır.
    """

    def detect(
        self, crop: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        """
        Araç crop'unda plaka tespit eder.

        Args:
            crop: BGR formatında araç kesimi

        Returns:
            (plate_bgr, [x1,y1,x2,y2]) veya (None, None)
            plate_bgr: Perspektif düzeltilmiş plaka crop'u
            bbox: crop-yerel koordinatlar
        """
        if crop is None or crop.size == 0:
            return None, None

        try:
            return self._detect_impl(crop)
        except Exception as e:
            logger.debug(f"CV plaka tespiti hatasi: {e}")
            return None, None

    def _detect_impl(
        self, crop: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        """İç implementasyon — hata yönetimi dışarıda."""

        # ── Adım 1: Ön işleme ──
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Bilateral filter: kenar koruyarak gürültü azaltır
        # d=11 → filtre çapı, sigmaColor=17, sigmaSpace=17
        # Plaka kenarlarını bulanıklaştırmadan arka planı yumuşatır
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)

        # ── Adım 2: Kenar tespiti (Sobel) ──
        # X ve Y yönünde gradyan hesapla
        # Plaka kenarları güçlü yatay ve dikey gradyanlar üretir
        sobelx = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
        sobel = cv2.convertScaleAbs(sobelx) + cv2.convertScaleAbs(sobely)

        # Otsu threshold — otomatik eşik belirleme
        _, thresh = cv2.threshold(
            sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Morfolojik kapama — kenar boşluklarını doldur
        # 21×7 kernel: yatay yönde agresif kapama → plaka dikdörtgenini birleştirir
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # ── Adım 3: Kontur bul ──
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ── Adım 4–5: Aspect ratio ve alan filtreleme ──
        candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if h == 0 or w == 0:
                continue

            aspect = w / h
            area = w * h

            # Türk plakası oranı: 520/110 ≈ 4.7
            # 3.5–5.5 aralığı perspektif toleransı sağlar
            if not (PLATE_ASPECT_MIN < aspect < PLATE_ASPECT_MAX):
                continue

            # Alan filtresi — çok küçük veya çok büyük adayları ele
            if not (PLATE_MIN_AREA < area < PLATE_MAX_AREA):
                continue

            # Genişlik oranı filtresi — Bumper ızgarası gibi arabanın çok büyük kısmını kaplayan blokları ele
            if (w / crop.shape[1]) > PLATE_MAX_WIDTH_RATIO:
                continue

            candidates.append((cnt, x, y, w, h, area))

        if not candidates:
            return None, None

        # En büyük aday genellikle en doğrusudur
        # (daha küçükler genelde parçalı kontur artıkları)
        candidates.sort(key=lambda c: c[5], reverse=True)
        cnt, x, y, w, h, _ = candidates[0]

        # ── Adım 6: Perspektif düzeltme ──
        plate_img = self._perspective_correct(crop, cnt, x, y, w, h)
        bbox = [x, y, x + w, y + h]

        return plate_img, bbox

    def _perspective_correct(
        self,
        crop: np.ndarray,
        contour: np.ndarray,
        x: int, y: int, w: int, h: int,
    ) -> np.ndarray:
        """
        4 noktalı kontur bulursa perspektif düzeltme uygular.
        Bulamazsa basit dikdörtgen crop döndürür.

        Perspektif düzeltme neden önemli:
        Kamera açısı nedeniyle plaka yamuk görünebilir.
        Düzeltme OCR doğruluğunu %15-20 artırır.
        """
        # Kontur yaklaşımı — 4 nokta elde etmeye çalış
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            # 4 noktalı çokgen bulundu → perspektif dönüşümü uygula
            pts = _order_points(approx.reshape(4, 2))

            # Hedef boyutlar — kaynak dörtgenin en-boy ölçüleri
            target_w = max(
                int(np.linalg.norm(pts[0] - pts[1])),
                int(np.linalg.norm(pts[2] - pts[3])),
            )
            target_h = max(
                int(np.linalg.norm(pts[0] - pts[3])),
                int(np.linalg.norm(pts[1] - pts[2])),
            )

            if target_w > 10 and target_h > 5:
                dst = np.array(
                    [
                        [0, 0],
                        [target_w - 1, 0],
                        [target_w - 1, target_h - 1],
                        [0, target_h - 1],
                    ],
                    dtype=np.float32,
                )

                M = cv2.getPerspectiveTransform(
                    pts.astype(np.float32), dst
                )
                warped = cv2.warpPerspective(
                    crop, M, (target_w, target_h)
                )
                return warped

        # Fallback: basit dikdörtgen crop
        # BUG FIX: Önceki if/else her iki branch'ta aynı şeyi yapıyordu.
        # Tek bir güvenli slice + copy yeterli.
        y2_safe = min(y + h, crop.shape[0])
        x2_safe = min(x + w, crop.shape[1])
        plate_crop = crop[y:y2_safe, x:x2_safe]
        if plate_crop.size == 0:
            return crop.copy()  # Son çare: tüm crop'u döndür
        return plate_crop.copy()
