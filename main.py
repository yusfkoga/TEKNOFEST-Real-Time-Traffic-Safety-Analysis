"""
TEKNOFEST 2026 — 5G & Yapay Zeka ile Akıllı Yol Güvenliği.

Ana pipeline modülü — asyncio tabanlı video analiz sistemi.
Tüm inference thread pool'larda çalışır, main thread asla bloklanmaz.

Mimari:
    Capture Thread       → cv2.VideoCapture, asyncio.Queue'ya frame yazar
    Detection Worker     → Tier-1 araç tespiti (ThreadPoolExecutor)
    Crop Analysis Worker → Tier-2a + 2b paralel (ThreadPoolExecutor)
    OCR Worker           → Ayrı ThreadPoolExecutor, rate-limited

Video vs Canlı Mod:
    Video dosyası  → Frame drop YOK, büyük buffer, kaynak FPS korunur
    Webcam/RTSP    → Frame drop AKTİF, düşük latency, küçük buffer

Kullanım:
    python main.py --source video.mp4
    python main.py --source video.mp4 --no-display
    python main.py --source video.mp4 --save-output
    python main.py --source 0  (webcam)
"""

import argparse
import asyncio
import logging
import os
import signal
import time

# ── ONNX Runtime GPU DLL FIX (Windows) ──
# ONNX Runtime provider DLL'leri (cublas64_12.dll vb.) Windows LoadLibrary
# ile yüklenir — bu PATH'e bakar, os.add_dll_directory'e değil.
# PyTorch kendi lib klasöründe bu DLL'leri barındırır; PATH'e ekliyoruz.
import site
for _sp in site.getsitepackages():
    _torch_lib = os.path.join(_sp, "torch", "lib")
    if os.path.isdir(_torch_lib):
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
        break

# ── CPU THREAD LIMITS (NMS TIMEOUT FIX) ──
# YOLO NMS için 4 thread yeterli — tüm çekirdekleri bağlamasın.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# OpenMP Kütüphane Çakışması Koruması (YOLO PyTorch + ONNX Runtime yan yana çalışınca)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# MediaPipe / TFLite C++ seviyesi uyarılarını sustur
os.environ["GLOG_minloglevel"] = "3"       # Linux/Mac
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # TFLite (Windows dahil)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # oneDNN info mesajlarını sustur
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# BUG FIX #1: sys.path'i import'lardan ÖNCE ayarla.
# Aksi hâlde aynı dizindeki modüller (config, tracker vb.)
# 'python main.py' ile çalıştırıldığında bulunamayabilir.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from config import (
    ANALYSIS_WORKERS,
    CABIN_EVERY_N,
    CAPTURE_QUEUE_SIZE,
    CROP_PADDING,
    DEAD_TRACK_CLEANUP_INTERVAL,
    DETECTION_WORKERS,
    FACE_MESH_EVERY_N,
    FACE_MESH_PERCLOS_THRESHOLD,
    FACE_MESH_PERCLOS_WINDOW,
    MAX_FRAME_WIDTH,
    OCR_QUEUE_SIZE,
    OCR_RATE_LIMIT_FRAMES,
    OCR_WORKERS,
    PLATE_CROP_TOP_RATIO,
    PLATE_EVERY_N,
    VIDEO_QUEUE_SIZE,
)
from criticality import compute_tier1_base, compute_risk, get_trigger_reasons
from detector import (
    CabinAnalyzer, FaceMeshAnalyzer, PoseAnalyzer, VehicleDetector,
    create_plate_detector, refine_plate_contour,
)
from ocr_worker import OCRWorker
from qod_manager import QoDManager
from tracker import TrackManager
from utils import FPSCounter, crop_with_padding
from visualizer import draw

logger = logging.getLogger("pipeline")


# ══════════════════════════════════════════════════════
# Logging Kurulumu
# ══════════════════════════════════════════════════════


def setup_logging() -> None:
    """
    Structured logging kurulumu.
    Tüm pipeline modülleri bu formatı kullanır.
    """
    fmt = "%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Ultralytics loglarını sustur — kendi loglarımız yeterli
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    # ONNX Runtime loglarını sustur
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
    # MediaPipe / TFLite / absl / matplotlib loglarını sustur
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ══════════════════════════════════════════════════════
# Argüman Ayrıştırma
# ══════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    """Komut satırı argümanlarını ayrıştırır."""
    parser = argparse.ArgumentParser(
        description="TEKNOFEST 2026 — 5G & AI Akilli Yol Guvenligi Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ornekler:\n"
            "  python main.py --source video.mp4\n"
            "  python main.py --source 0 --no-display\n"
            "  python main.py --source video.mp4 --save-output\n"
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Video dosyasi yolu veya kamera indeksi (0, 1, ...)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Goruntuleme penceresini devre disi birak",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Islenmis videoyu output.mp4 olarak kaydet",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════
# Kaynak Analizi — Video mi Canlı mı?
# ══════════════════════════════════════════════════════


def analyze_source(source: str) -> dict:
    """
    Video kaynağını analiz eder: dosya mı, canlı akış mı?
    FPS, çözünürlük, toplam frame gibi metadata'ları döndürür.

    Returns:
        {
            "is_live": bool,
            "source_fps": float,
            "width": int,
            "height": int,
            "total_frames": int,
            "duration_sec": float,
        }
    """
    info = {
        "is_live": False,
        "source_fps": 30.0,
        "width": 0,
        "height": 0,
        "total_frames": 0,
        "duration_sec": 0.0,
    }

    # Sayısal ise kamera indeksi → canlı
    try:
        int(source)
        info["is_live"] = True
        return info
    except (ValueError, TypeError):
        pass

    # Dosya mevcut mu?
    if not os.path.isfile(source):
        # RTSP/HTTP akışı olabilir
        info["is_live"] = True
        return info

    # Video dosyası — metadata oku
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        info["source_fps"] = fps if 1 < fps < 120 else 30.0
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if info["source_fps"] > 0 and info["total_frames"] > 0:
            info["duration_sec"] = info["total_frames"] / info["source_fps"]
        cap.release()

    return info


# ══════════════════════════════════════════════════════
# Capture Thread
# ══════════════════════════════════════════════════════


async def capture_frames(
    source, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop,
    is_live: bool = True,
) -> None:
    """
    Video kaynağından frame okur ve asyncio.Queue'ya yazar.
    Blocking cv2.VideoCapture.read() thread pool'da çalışır.

    Mod farkları:
        Canlı  : Queue doluysa eski frame atılır → düşük latency
        Video  : Frame ASLA atılmaz → her frame işlenir, output doğru sürede
    """
    # Kaynak sayısal ise kamera indeksi olarak yorumla
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Video kaynagi acilamadi: {source}")
        # Sentinel gönder — main loop'un takılmasını önle
        await queue.put(None)
        return

    # Video bilgilerini logla
    fps_src = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resize sonrası gerçek çözünürlüğü hesapla (log için)
    if MAX_FRAME_WIDTH and width > MAX_FRAME_WIDTH:
        proc_scale = MAX_FRAME_WIDTH / width
        proc_w, proc_h = MAX_FRAME_WIDTH, int(height * proc_scale)
    else:
        proc_w, proc_h = width, height

    logger.info(
        f"Video acildi: {width}x{height} @ {fps_src:.1f}fps | "
        f"Toplam frame: {total if total > 0 else 'canli'} | "
        f"Mod: {'CANLI' if is_live else 'DOSYA'}"
    )
    if proc_w != width:
        logger.info(
            f"Frame resize aktif: {width}x{height} -> {proc_w}x{proc_h} "
            f"(MAX_FRAME_WIDTH={MAX_FRAME_WIDTH})"
        )

    try:
        while True:
            # Blocking read → thread pool'da çalıştır
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                logger.info("Video akisi sona erdi")
                break

            # ── Capture anında resize (4K+ için bellek tasarrufu) ──
            if MAX_FRAME_WIDTH and frame.shape[1] > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / frame.shape[1]
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(
                    frame, (MAX_FRAME_WIDTH, new_h), interpolation=cv2.INTER_AREA
                )

            if is_live:
                # Canlı mod: Queue doluysa eski frame'i at — düşük latency
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await queue.put(frame)
            else:
                # Video modu: ASLA frame atma — tüm frame'ler işlenmeli
                # queue.put() doğal backpressure sağlar:
                # eğer queue doluysa, capture işleme hızını bekler
                await queue.put(frame)

    except asyncio.CancelledError:
        logger.info("Capture gorevi iptal edildi")
        raise
    finally:
        cap.release()
        # Sentinel: main loop'a "artık frame yok" sinyali gönder
        try:
            await queue.put(None)
        except asyncio.CancelledError:
            pass
        logger.info("Video kaynagi serbest birakildi")


# ══════════════════════════════════════════════════════
# OCR Consumer
# ══════════════════════════════════════════════════════


async def ocr_consumer(
    queue: asyncio.Queue,
    ocr_engine: OCRWorker,
    tracker: TrackManager,
    pool: ThreadPoolExecutor,
) -> None:
    """
    OCR Worker — plaka crop'larını kuyruktan alıp işler.
    Ayrı ThreadPoolExecutor'da çalışır.
    Rate limiting tracker'daki plate_last_ocr_frame ile sağlanır.
    """
    loop = asyncio.get_running_loop()
    while True:
        try:
            track_id, plate_img, frame_idx = await queue.get()

            # OCR inference'ı thread pool'da çalıştır — ASLA main thread'de
            text, confidence = await loop.run_in_executor(
                pool, ocr_engine.read_plate, plate_img
            )

            # Sonucu merkezi track state'e yaz
            track = tracker.get_track(track_id)
            if track and text:
                track.plate_text = text
                track.plate_confidence = confidence
                track.plate_last_ocr_frame = frame_idx
                logger.info(
                    f"PLAKA_OKUMA | track={track_id} | "
                    f"text={text} | conf={confidence:.2f}"
                )
            elif track:
                # OCR başarısız — rate limit güncelle ki sürekli denemesin
                track.plate_last_ocr_frame = frame_idx
                logger.debug(
                    f"OCR_BOS | track={track_id} | "
                    f"frame={frame_idx} | img_shape={plate_img.shape}"
                )

            queue.task_done()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"OCR worker hatasi: {e}", exc_info=True)


# ══════════════════════════════════════════════════════
# Araç Analizi (Tier-2a + Tier-2b Paralel)
# ══════════════════════════════════════════════════════


async def process_vehicle(
    track,
    crop,
    crop_offset_x: int,
    crop_offset_y: int,
    frame_idx: int,
    cabin_analyzer: CabinAnalyzer,
    plate_detector,
    pose_analyzer: PoseAnalyzer,
    face_mesh_analyzer: FaceMeshAnalyzer,
    ocr_queue: asyncio.Queue,
    pool: ThreadPoolExecutor,
    pose_pool: ThreadPoolExecutor,
    is_fresh_detection: bool = True,
    use_sahi: bool = False,
) -> None:
    """
    Tek bir araç için Tier-2a ve Tier-2b analizini paralel çalıştırır.

    Her araç için asyncio.gather() ile Tier-2a (kabin) ve
    Tier-2b (plaka) aynı crop üzerinde eşzamanlı çalışır.
    3 araç varsa 3 ayrı process_vehicle coroutine'i paralel koşar.

    use_sahi=True ise kabin analizinde SAHI ile tile'lı inference yapılır
    (QoD tetiklendiğinde, temiz görüntü + SAHI = doğruluk artışı).

    Plaka tespiti sadece taze detection frame'lerinde yapılır.
    Cache frame'lerinde bbox kayması yanlış pozitif üretir.
    """
    loop = asyncio.get_running_loop()
    tasks = []
    task_labels = []

    # ── Tier-2a: Kabin analizi (her CABIN_EVERY_N frame'de) ──
    # Sadece üst %60 (cam/kabin bölgesi). Alt %40 tampon/plaka/tekerlek — gereksiz.
    # Daha küçük image → YOLO daha hızlı çalışır.
    if frame_idx % CABIN_EVERY_N == 0:
        cabin_h = int(crop.shape[0] * 0.60)
        cabin_crop = crop[:cabin_h, :]
        # Tier-2 + QoD aktif → SAHI ile derin doğrulama
        # Normal → standart detect (hızlı tarama)
        detect_fn = cabin_analyzer.detect_with_sahi if use_sahi else cabin_analyzer.detect
        tasks.append(
            loop.run_in_executor(pool, detect_fn, cabin_crop)
        )
        task_labels.append("cabin")

    # ── Tier-2b: Plaka tespiti (her PLATE_EVERY_N frame'de) ──
    # Plaka okunmuşsa → YOLO inference ATLA (GPU tasarrufu).
    # EMA bbox zaten track'ta korunuyor, görselleştirme devam eder.
    plate_already_read = (
        track.plate_text != "" and track.plate_confidence >= 0.75
    )
    if (
        frame_idx % PLATE_EVERY_N == 0
        and is_fresh_detection
        and not plate_already_read  # Okunmuş plaka → YOLO boşa çalışmasın
    ):
        # Sadece crop'un alt kısmında plaka ara
        # Plakalar aracın alt %55'inde bulunur — üst yarı cam/gökyüzü
        crop_h = crop.shape[0]
        plate_top = int(crop_h * PLATE_CROP_TOP_RATIO)
        lower_crop = crop[plate_top:, :]

        if lower_crop.size > 0:
            # Plaka tespiti: direkt YOLO detect (SAHI gereksiz overhead)
            # SAHI 160px tile'lar → plaka tile sınırında kesilir → miss.
            # Direkt 640px inference plaka bulmaya yeterli.
            tasks.append(
                loop.run_in_executor(pool, plate_detector.detect, lower_crop)
            )
            task_labels.append("plate")

    if not tasks:
        return

    # Tier-2a ve Tier-2b paralel çalışsın
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for label, result in zip(task_labels, results):
        if isinstance(result, Exception):
            logger.error(f"Tier-2 hatasi ({label}): {result}")
            continue

        if label == "cabin":
            # Kabin nesnelerini crop-yerel → frame-mutlak koordinatlara çevir
            cabin_objects_abs = []
            for obj in result:
                abs_bbox = [
                    obj["bbox"][0] + crop_offset_x,
                    obj["bbox"][1] + crop_offset_y,
                    obj["bbox"][2] + crop_offset_x,
                    obj["bbox"][3] + crop_offset_y,
                ]
                cabin_objects_abs.append({**obj, "bbox": abs_bbox})
            track.cabin_objects = cabin_objects_abs

            # Kabin tespiti logla (debug)
            if cabin_objects_abs:
                obj_summary = ", ".join(
                    f"{o['label']}({o['confidence']:.0%})" for o in cabin_objects_abs
                )
                logger.debug(
                    f"CABIN | track={track.track_id[:8]} | {obj_summary}"
                )

            # ── Pose & Face Mesh: Fire-and-forget async ──
            # Pipeline BEKLEMEZ — arka planda çalışır, bitince track güncellenir.
            persons = [o for o in result if o.get("class_id") == 0]
            run_pose = (
                persons
                and pose_analyzer.ready
                and frame_idx % (CABIN_EVERY_N * 2) == 0
            )
            run_face = (
                persons
                and face_mesh_analyzer.ready
                and frame_idx % (CABIN_EVERY_N * FACE_MESH_EVERY_N) == 0
            )

            # Ortak küçük crop — pose VEYA face mesh çalışacaksa TEK SEFER hesapla
            # 1920x1080 → 320x180 (~30x daha az piksel kopyalanır)
            crop_small = None
            if run_pose or run_face:
                h, w = crop.shape[:2]
                _max = 320
                if w > _max:
                    _s = _max / w
                    crop_small = cv2.resize(crop, (int(w * _s), int(h * _s)), interpolation=cv2.INTER_AREA)
                else:
                    crop_small = crop.copy()

            if run_pose:
                track_ref = track

                async def _pose_bg(t=track_ref, c=crop_small):
                    r = await loop.run_in_executor(pose_pool, pose_analyzer.analyze, c)
                    t.phone_detected = r["phone_call"]
                    t.phone_score = r["score"]
                    t.cigarette_detected = r["smoking"]
                    t.cigarette_score = r["smoking_score"]
                    if r["phone_call"]:
                        logger.info(
                            f"POSE_TELEFON | track={t.track_id[:8]} | "
                            f"score={r['score']:.2f} | side={r['side']}"
                        )
                    if r["smoking"]:
                        logger.info(
                            f"POSE_SIGARA | track={t.track_id[:8]} | "
                            f"score={r['smoking_score']:.2f} | side={r['smoking_side']}"
                        )

                asyncio.create_task(_pose_bg())

            elif not persons:
                # Kişi bulunamadı → flag'leri sıfırla (false positive önlemi)
                track.phone_detected = False
                track.phone_score = 0.0
                track.cigarette_detected = False
                track.cigarette_score = 0.0
            # persons var ama pose frame'i değil → önceki değeri koru

            if run_face:
                track_ref2 = track

                async def _face_bg(t=track_ref2, c=crop_small):
                    r = await loop.run_in_executor(
                        pose_pool, face_mesh_analyzer.analyze, c
                    )
                    t.ear_left = r["ear_left"]
                    t.ear_right = r["ear_right"]

                    # PERCLOS: deque otomatik maxlen ile eski kayıtları atar
                    t._eye_closed_history.append(r["eye_closed"])
                    hist_len = len(t._eye_closed_history)
                    if hist_len >= FACE_MESH_PERCLOS_WINDOW // 2:
                        t.perclos = sum(t._eye_closed_history) / hist_len
                        t.drowsy_detected = t.perclos > FACE_MESH_PERCLOS_THRESHOLD
                    else:
                        t.perclos = 0.0
                        t.drowsy_detected = False

                    # MAR / esneme
                    t.mar = r["mar"]
                    t.yawn_detected = r["yawn"]

                    if t.drowsy_detected:
                        logger.info(
                            f"FACE_UYUKLAMA | track={t.track_id[:8]} | "
                            f"PERCLOS={t.perclos:.2f} | EAR={r['ear_avg']:.3f}"
                        )
                    if t.yawn_detected:
                        logger.info(
                            f"FACE_ESNEME | track={t.track_id[:8]} | "
                            f"MAR={r['mar']:.3f}"
                        )

                asyncio.create_task(_face_bg())

        elif label == "plate":
            plate_img, plate_bbox_rel = result

            if plate_img is not None and plate_bbox_rel is not None:
                # Plaka bbox'ını frame-mutlak koordinatlara çevir
                # plate_top offset: lower_crop üst sınırı
                crop_h = crop.shape[0]
                plate_top = int(crop_h * PLATE_CROP_TOP_RATIO)
                new_bbox = [
                    plate_bbox_rel[0] + crop_offset_x,
                    plate_bbox_rel[1] + crop_offset_y + plate_top,
                    plate_bbox_rel[2] + crop_offset_x,
                    plate_bbox_rel[3] + crop_offset_y + plate_top,
                ]

                # EMA smoothing — zıplamayı önle
                # alpha=0.3: yeni değere %30 ağırlık, eskiye %70 (daha yumuşak)
                if track.plate_bbox_abs is not None:
                    a = 0.3
                    track.plate_bbox_abs = [
                        int(a * new_bbox[i] + (1 - a) * track.plate_bbox_abs[i])
                        for i in range(4)
                    ]
                else:
                    track.plate_bbox_abs = new_bbox

                # Oriented bbox — plaka açılı duruyorsa sıkı sarar
                # plate_bbox_rel lower_crop koordinatında → plate_top ekle
                plate_bbox_in_crop = [
                    plate_bbox_rel[0],
                    plate_bbox_rel[1] + plate_top,
                    plate_bbox_rel[2],
                    plate_bbox_rel[3] + plate_top,
                ]
                corners = refine_plate_contour(crop, plate_bbox_in_crop)
                if corners is not None:
                    # crop-yerel → frame-mutlak koordinata çevir
                    abs_corners = corners.copy()
                    abs_corners[:, 0] += crop_offset_x
                    abs_corners[:, 1] += crop_offset_y
                    # EMA smoothing corners için de
                    if track.plate_corners_abs is not None:
                        a = 0.3
                        track.plate_corners_abs = np.intp(
                            a * abs_corners + (1 - a) * track.plate_corners_abs
                        )
                    else:
                        track.plate_corners_abs = abs_corners
                # corners None ise önceki değeri koru (jitter önlemi)

                # OCR kuyruğuna gönder (rate limit + kalite kontrolü)
                # Plaka zaten güvenle okunduysa OCR atla — sadece bbox takibi devam eder
                should_ocr = (
                    not plate_already_read
                    and frame_idx - track.plate_last_ocr_frame
                    >= OCR_RATE_LIMIT_FRAMES
                )

                if should_ocr:
                    try:
                        ocr_queue.put_nowait(
                            (track.track_id, plate_img, frame_idx)
                        )
                    except asyncio.QueueFull:
                        pass  # Kuyruk doluysa at — sonraki frame'de tekrar dener
            else:
                # Plaka tespit edilemedi:
                # - Okunmuş plaka → bbox koru (smooth takip devam eder)
                # - Okunmamış plaka → bbox temizle (ghost box önlemi)
                if track.plate_text == "":
                    track.plate_bbox_abs = None


# ══════════════════════════════════════════════════════
# Ana Pipeline Döngüsü
# ══════════════════════════════════════════════════════


async def run_pipeline(args: argparse.Namespace) -> None:
    """
    Ana asenkron pipeline döngüsü.

    Akış:
        Frame al → Tier-1 → Tracker güncelle → Tier-2a+2b (paralel)
        → OCR (async) → Kritiklik hesapla → QoD stub → Görselleştir

    Video modu:
        - Tüm frame'ler işlenir (hiçbiri atılmaz)
        - VideoWriter kaynak FPS'i kullanır
        - Output süresi = kaynak süresi
    """
    loop = asyncio.get_running_loop()

    # ── Banner ──
    logger.info("=" * 60)
    logger.info("  TEKNOFEST 2026 — Akilli Yol Guvenligi Pipeline")
    logger.info("  5G & Yapay Zeka ile Gercek Zamanli Analiz")
    logger.info("=" * 60)

    # ── Kaynak Analizi ──
    source_info = analyze_source(args.source)
    is_live = source_info["is_live"]
    source_fps = source_info["source_fps"]

    logger.info(
        f"Kaynak modu: {'CANLI' if is_live else 'VIDEO DOSYASI'} | "
        f"FPS: {source_fps:.1f} | "
        f"Sure: {source_info['duration_sec']:.1f}s | "
        f"Frame: {source_info['total_frames']}"
    )

    # ── Model Yükleme ──
    logger.info("Modeller yukleniyor...")
    vehicle_detector = VehicleDetector()
    cabin_analyzer = CabinAnalyzer()
    plate_detector = create_plate_detector()
    pose_analyzer = PoseAnalyzer()
    face_mesh_analyzer = FaceMeshAnalyzer()
    ocr_engine = OCRWorker()
    tracker = TrackManager()
    qod_manager = QoDManager()
    logger.info("Tum modeller yuklendi (QoD Manager hazir)")

    # ── Queue'lar ──
    # Video modu: büyük buffer (frame kaybı yok)
    # Canlı mod: küçük buffer (düşük latency)
    queue_size = CAPTURE_QUEUE_SIZE if is_live else VIDEO_QUEUE_SIZE
    capture_queue = asyncio.Queue(maxsize=queue_size)
    ocr_queue = asyncio.Queue(maxsize=OCR_QUEUE_SIZE)

    # ── Thread Pool'lar ──
    detect_pool = ThreadPoolExecutor(
        max_workers=DETECTION_WORKERS,
        thread_name_prefix="detect",
    )
    analysis_pool = ThreadPoolExecutor(
        max_workers=ANALYSIS_WORKERS,
        thread_name_prefix="analysis",
    )
    ocr_pool = ThreadPoolExecutor(
        max_workers=OCR_WORKERS,
        thread_name_prefix="ocr",
    )
    pose_pool = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="pose",
    )

    # ── Arka Plan Görevleri ──
    capture_task = asyncio.create_task(
        capture_frames(args.source, capture_queue, loop, is_live=is_live)
    )
    ocr_task = asyncio.create_task(
        ocr_consumer(ocr_queue, ocr_engine, tracker, ocr_pool)
    )

    frame_idx = 0
    fps_counter = FPSCounter()
    video_writer = None

    logger.info(f"Pipeline baslatildi | Kaynak: {args.source}")
    logger.info(f"Display: {'KAPALI' if args.no_display else 'ACIK'}")
    logger.info(f"Kayit: {'ACIK' if args.save_output else 'KAPALI'}")
    logger.info(f"Queue boyutu: {queue_size} ({'canli' if is_live else 'video'})")
    logger.info("-" * 60)

    try:
        while True:
            # ── Frame Al ──
            try:
                frame = await asyncio.wait_for(
                    capture_queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                if capture_task.done():
                    logger.info(
                        "Capture tamamlandi, pipeline durduruluyor"
                    )
                    break
                continue

            # Sentinel kontrolü — capture bitti
            if frame is None:
                logger.info("Sentinel alindi, tum frame'ler islendi")
                break

            frame_idx += 1
            fps_counter.tick()

            # ── Tier-1: Araç Tespiti + BoTSORT Tracking ──
            # BoTSORT her frame'de çalışmalı (Kalman filtre sürekliliği)
            # model.track(persist=True) ile ID ataması yapılır
            tracked_detections = await loop.run_in_executor(
                detect_pool, vehicle_detector.track, frame
            )
            is_fresh_detection = True  # BoTSORT her frame taze

            # ── TrackManager Güncelle ──
            tracker.update(tracked_detections, frame_idx)

            # ── Tier-2 + Kritiklik: Tek Döngüde Birleşik Analiz ──
            # (Önceden iki ayrı döngüdeydi → compute_tier1_base iki kez çağrılıyordu)
            analysis_tasks = []
            qod_count = 0
            current_time = time.time()
            qod_tasks = []  # QoD güncellemelerini topla, sonra batch olarak çalıştır

            for track in tracker.get_active_tracks():
                if not track.bbox_history:
                    continue

                # Tier-1 Baz Riski — TEK SEFER hesapla
                tier1_base = compute_tier1_base(track, frame.shape)
                
                # Hysteresis tetikleme koşulu: Risk > 45
                can_trigger_t2 = (tier1_base >= 45.0)  
                
                # Eger Arac Tier-2'ye gidiyorsa:
                if can_trigger_t2:
                    bbox = track.bbox_history[-1]
                    crop, ox, oy = crop_with_padding(frame, bbox, CROP_PADDING)
                    if crop is not None:
                        track.last_t2_frame = frame_idx
                        task = process_vehicle(
                            track, crop, ox, oy, frame_idx,
                            cabin_analyzer, plate_detector, pose_analyzer,
                            face_mesh_analyzer,
                            ocr_queue, analysis_pool, pose_pool,
                            is_fresh_detection=is_fresh_detection,
                            use_sahi=can_trigger_t2,
                        )
                        analysis_tasks.append(task)

                # ── Kritiklik hesapla — KTK ağırlıklı (aynı döngüde, ek maliyet yok) ──
                risk_info = compute_risk(track.cabin_objects, track, tier1_base)
                track.criticality_score = risk_info["score"]

                # İhlal tespit edildiyse KTK detaylarını logla
                if risk_info.get("violations"):
                    reasons = get_trigger_reasons(risk_info)
                    for reason in reasons:
                        logger.warning(
                            f"KTK_IHLAL | track={track.track_id[:8]} | "
                            f"skor={risk_info['score']:.1f} | {reason}"
                        )

                if risk_info.get("force_qod_l") or risk_info.get("score", 0) >= 60:
                    qod_count += 1
                    # Sadece yüksek riskli track'lar için QoD güncelleme kuyruğuna ekle
                    qod_tasks.append(qod_manager.update(risk_info, current_time))
                else:
                    # Düşük risk → session kapatma mantığı için yine güncelle
                    # ama sadece aktif session varsa (gereksiz task oluşturmayı önle)
                    if qod_manager.current_profile is not None:
                        qod_tasks.append(qod_manager.update(risk_info, current_time))

            # N araç varsa N coroutine paralel çalışır (Batch Async)
            if analysis_tasks:
                await asyncio.gather(*analysis_tasks, return_exceptions=True)

            # QoD güncellemelerini toplu çalıştır (create_task spam'i yerine)
            if qod_tasks:
                await asyncio.gather(*qod_tasks, return_exceptions=True)

            # ── Görselleştirme ──
            if not args.no_display or args.save_output:
                vis_frame = draw(
                    frame, tracker, frame_idx, fps_counter.fps, qod_count
                )

                if not args.no_display:
                    cv2.imshow(
                        "TEKNOFEST 2026 - Akilli Yol Guvenligi",
                        vis_frame,
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:  # q veya ESC
                        logger.info("Kullanici cikis yapti (q/ESC)")
                        break

                if args.save_output:
                    if video_writer is None:
                        h, w = vis_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        # KAYNAK FPS kullan — hardcoded 25 DEĞİL!
                        writer_fps = source_fps if not is_live else 25.0
                        video_writer = cv2.VideoWriter(
                            "output.mp4", fourcc, writer_fps, (w, h)
                        )
                        logger.info(
                            f"Video kaydi baslatildi: {w}x{h} @ {writer_fps:.1f}fps"
                        )
                    video_writer.write(vis_frame)

            # ── Periyodik Temizlik ──
            removed = tracker.cleanup(frame_idx)
            if removed and frame_idx % DEAD_TRACK_CLEANUP_INTERVAL == 0:
                logger.info(
                    f"Temizlik: {removed} olu track silindi | "
                    f"Aktif: {tracker.active_count}"
                )

            # ── Periyodik İlerleme Logu ──
            if frame_idx % 30 == 0:
                total = source_info["total_frames"]
                pct = f" ({frame_idx/total:.0%})" if total > 0 else ""
                logger.info(
                    f"Ilerleme: frame={frame_idx}{pct} | "
                    f"FPS={fps_counter.fps:.1f} | "
                    f"arac={tracker.active_count}"
                )

    except KeyboardInterrupt:
        logger.info("Pipeline kullanici tarafindan durduruldu")

    finally:
        # ── Graceful Shutdown ──
        logger.info("-" * 60)
        logger.info("Pipeline kapatiliyor...")

        capture_task.cancel()
        ocr_task.cancel()

        # Task'ların bitmesini bekle
        for task in [capture_task, ocr_task]:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        detect_pool.shutdown(wait=False)
        analysis_pool.shutdown(wait=False)
        ocr_pool.shutdown(wait=False)
        pose_pool.shutdown(wait=False)

        if video_writer is not None:
            video_writer.release()
            logger.info(f"Video ciktisi kaydedildi: output.mp4 ({frame_idx} frame)")

        cv2.destroyAllWindows()

        logger.info("=" * 60)
        logger.info(f"  Toplam islenen frame: {frame_idx}")
        logger.info(f"  Son FPS: {fps_counter.fps:.1f}")
        logger.info(f"  Aktif track sayisi: {tracker.active_count}")
        logger.info("  Pipeline kapatildi")
        logger.info("=" * 60)


# ══════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════


def main() -> None:
    """Program giriş noktası."""
    setup_logging()
    args = parse_args()

    # Windows'ta SIGINT zaten KeyboardInterrupt ürettiği için
    # ek handler gerekmez; sadece Linux/Mac için eklenebilir.
    if sys.platform != "win32":
        def signal_handler(sig, _frame):
            logger.info(f"Sinyal alindi: {sig}")
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, signal_handler)

    # Ana async döngüyü başlat
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
