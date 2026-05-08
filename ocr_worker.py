"""
TEKNOFEST 2026 — OCR Pipeline (fast-plate-ocr).

fast-plate-ocr (CCT modeli) tabanlı plaka metin okuma modülü.
Plakaya özel hafif ONNX model — PaddleOCR'ın 3 aşamalı pipeline'ına
(det+cls+rec) gerek kalmadan tek modelle doğrudan karakter okur.

OCR, main thread'de ASLA çalışmaz:
    - Ayrı ThreadPoolExecutor'da çalışır
    - Rate limiting: aynı track_id için min N frame arayla
    - asyncio.Queue üzerinden crop'lar alınır

PaddleOCR → fast-plate-ocr geçiş nedenleri:
    - PaddleOCR 3 model çalıştırır (det+cls+rec), biz sadece rec'e ihtiyaç duyuyoruz
    - PaddleOCR'ın det modeli crop'lanmış plakayı tekrar metin tespiti yapıp
      harf-harf bölebiliyordu — en büyük bug kaynağı buydu
    - fast-plate-ocr ~2-5ms/plaka (PaddleOCR ~100-300ms)
    - ONNX Runtime → CUDA/TensorRT desteği, ~20MB VRAM (PaddleOCR ~400MB)
"""

import logging
import re
from typing import Tuple

import numpy as np

from config import (
    OCR_MODEL_NAME,
    PLATE_REGEX,
)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════
# Türk Plaka Karakter Dönüşüm Tabloları
# ══════════════════════════════════════════════════════
# OCR'ın sık yaptığı O↔0, I↔1, B↔8 hatalarını Türk plaka
# formatına göre düzeltmek için kullanılır.

_NUM_TO_CHAR = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z', '6': 'G', '4': 'A'}
_CHAR_TO_NUM = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '2', 'G': '6', 'A': '4'}


class OCRWorker:
    """
    fast-plate-ocr tabanlı plaka okuyucu.

    Lazy initialization: ONNX modeli ilk kullanımda yüklenir.
    Bu sayede pipeline başlatma süresi kısalır.

    Pipeline:
        1. Plaka crop'u alınır (YOLO detector'dan)
        2. fast-plate-ocr modeline verilir → doğrudan karakter dizisi döner
        3. Türk plaka heuristikleriyle düzeltilir
        4. Format validasyonundan geçirilir
    """

    def __init__(self):
        self._plate_regex = re.compile(PLATE_REGEX)

        # Eager init — TensorRT engine compile startup'ta olsun,
        # pipeline ortasında 10-30sn bloklanmayı önler
        logger.info(f"fast-plate-ocr baslatiliyor: {OCR_MODEL_NAME}")
        from fast_plate_ocr import LicensePlateRecognizer

        self._ocr = LicensePlateRecognizer(OCR_MODEL_NAME, device="cuda")
        logger.info("fast-plate-ocr hazir (ONNX Runtime)")

        # Warmup — TensorRT engine compile'i tetikle
        dummy = np.zeros((50, 200, 3), dtype=np.uint8)
        try:
            self._ocr.run(dummy)
            logger.info("fast-plate-ocr warmup tamamlandi")
        except Exception:
            pass  # Bos dummy hata verirse sorun degil

    def read_plate(
        self, plate_img: np.ndarray
    ) -> Tuple[str, float]:
        """
        Plaka görüntüsünden metin okur.

        Args:
            plate_img: BGR formatında plaka crop'u

        Returns:
            (text, confidence) — geçerli format bulunamazsa ("", 0.0)
        """
        if plate_img is None or plate_img.size == 0:
            return "", 0.0

        try:
            ocr = self._ocr

            # fast-plate-ocr doğrudan crop alır, return_confidence=True ile
            # [PlatePrediction(plate=str, char_probs=ndarray), ...] döner
            results = ocr.run(plate_img, return_confidence=True)

            if not results:
                return "", 0.0

            # İlk (ve genelde tek) sonucu al
            prediction = results[0]
            raw_text = prediction.plate
            # char_probs: her karakter için olasılık dizisi → ortalaması = genel güven
            confidence = float(prediction.char_probs.mean()) if prediction.char_probs is not None else 0.0

            logger.debug(f"OCR_RAW | text='{raw_text}' conf={confidence:.2f}")

            if confidence < 0.15:
                return "", 0.0

            # Temizle — sadece alfanümerik
            cleaned = raw_text.strip().upper()
            cleaned = ''.join(c for c in cleaned if c.isalnum())

            if not cleaned:
                return "", 0.0

            # ── Türk plaka heuristik düzeltme ──
            formatted = self._format_turkish_plate(cleaned)
            logger.debug(f"OCR_FORMAT | '{cleaned}' -> '{formatted}'")

            # ── Validasyon ──
            if self._validate_plate(formatted):
                logger.info(f"OCR_VALID | text='{formatted}' conf={confidence:.2f}")
                return formatted, confidence

            logger.info(f"OCR_REJECTED | text='{formatted}' conf={confidence:.2f}")
            return "", 0.0

        except Exception as e:
            logger.error(f"OCR hatasi: {e}", exc_info=True)
            return "", 0.0

    @staticmethod
    def _format_turkish_plate(text: str) -> str:
        """
        Türk plaka format heuristikleri.

        Karakter tipi geçişleriyle böler (digit→letter, letter→digit).
        OCR'ın sık yaptığı O/0, I/1, B/8, S/5 hatalarını pozisyona göre düzeltir.

        ÖNCEKİ BUG: Greedy regex ([A-Z0-9]{1,3}) "34TC8532"'u
        ["34","TC8","532"] olarak bölerdi → 8→B → "34 TCB 532" çöp!

        YENİ: re.findall(r'[A-Z]+|\\d+') ile karakter tipine göre böler:
        "34TC8532" → ["34","TC","8532"] (doğru)
        """
        raw = text.upper().replace(" ", "")

        # OCR Türk plakasının mavi bölgesindeki TR yazısını kazara okuyabilir
        if raw.startswith("TR") and len(raw) >= 6:
            raw = raw[2:]

        # Çok kısa veya çok uzun → dokunma
        if len(raw) < 5 or len(raw) > 10:
            return text

        # ── Strateji 1: Karakter tipi geçişleriyle böl ──
        # "34TC8532" → ["34", "TC", "8532"] ✓
        # "06AB123"  → ["06", "AB", "123"]  ✓
        parts = re.findall(r'[A-Z]+|\d+', raw)

        if len(parts) == 3:
            p1, p2, p3 = parts
            # Yapı kontrolü: 1-2 rakam, 1-3 harf, 2-4 rakam
            if (1 <= len(p1) <= 2 and p1.isdigit() and
                1 <= len(p2) <= 3 and p2.isalpha() and
                2 <= len(p3) <= 4 and p3.isdigit()):
                code = int(p1.zfill(2))
                if 1 <= code <= 81:
                    return f"{p1.zfill(2)} {p2} {p3}"

        # ── Strateji 2: OCR karıştırdıysa (O↔0, I↔1) pozisyon bazlı düzelt ──
        # İlk 2 karakter → rakam yap, ortadakiler → harf yap, sondakiler → rakam yap
        if len(raw) < 5 or len(raw) > 10:
            return text

        # İlk 2 karakter: il kodu (RAKAM olmalı)
        p1_raw = raw[:2]
        p1 = "".join(_CHAR_TO_NUM.get(c, c) for c in p1_raw)

        # Ortadaki harfleri bul: 2. pozisyondan itibaren ilk rakama kadar
        rest = raw[2:]
        letter_end = 0
        for i, c in enumerate(rest):
            # Karakter harf veya harf olabilecek rakam mı?
            if c.isalpha() or (c in _NUM_TO_CHAR and i < 3):
                letter_end = i + 1
            else:
                break

        if letter_end == 0:
            letter_end = min(len(rest) - 2, 3)  # En az 2 rakam sonda kalmalı

        p2_raw = rest[:letter_end]
        p3_raw = rest[letter_end:]

        if not p2_raw or not p3_raw:
            return text

        p2 = "".join(_NUM_TO_CHAR.get(c, c) for c in p2_raw)
        p3 = "".join(_CHAR_TO_NUM.get(c, c) for c in p3_raw)

        # Son validasyon
        if p1.isdigit() and 1 <= int(p1) <= 81 and p2.isalpha() and p3.isdigit():
            return f"{p1} {p2} {p3}"

        return text

    def _validate_plate(self, text: str) -> bool:
        """
        Türk plaka format validasyonu.
        Hem tam format hem de kısmi eşleşmeyi kontrol eder.
        """
        normalized = " ".join(text.split())

        # Tam regex eşleşmesi
        if self._plate_regex.search(normalized):
            return True

        # Kısmi kabul: "XX YYY ZZZZ" yapısında mı?
        # Regex'e tam uymasa bile 2rakam+harf+rakam yapısı varsa kabul et
        parts = normalized.split()
        if len(parts) == 3:
            p1, p2, p3 = parts
            if (p1.isdigit() and 1 <= int(p1) <= 81 and
                p2.isalpha() and 1 <= len(p2) <= 3 and
                p3.isdigit() and 1 <= len(p3) <= 5):
                return True

        return False
