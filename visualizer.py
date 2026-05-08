"""
TEKNOFEST 2026 — Görselleştirme Modülü.

Pipeline mantığından tamamen ayrıdır.
Sadece çizim yapar, hiçbir analiz veya karar vermez.

Renkler (BGR):
    Normal araç  : (0, 255, 0)    yeşil
    Kabin nesnesi : (0, 165, 255)  turuncu
    Plaka bbox   : (255, 0, 255)  mor
    Kritik araç  : (0, 0, 255)    kırmızı

Sol üst HUD paneli:
    Frame sayısı, FPS, aktif araç sayısı, QoD durumu, uyarılar
"""

import logging
from typing import List

import cv2
import numpy as np

from config import (
    COLOR_CABIN_OBJECT,
    COLOR_PLATE_BBOX,
    COLOR_VEHICLE_CRITICAL,
    COLOR_VEHICLE_NORMAL,
    CRITICALITY_THRESHOLD,
    VIS_FONT_SCALE,
    VIS_THICKNESS_CRITICAL,
    VIS_THICKNESS_NORMAL,
)

logger = logging.getLogger(__name__)

# OpenCV font — Unicode desteklemez, ASCII kullanılır
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw(
    frame: np.ndarray,
    tracker,
    frame_idx: int,
    fps: float,
    qod_count: int,
) -> np.ndarray:
    """
    Frame üzerine tüm görselleştirme katmanlarını çizer.

    Args:
        frame: BGR formatında orijinal frame
        tracker: TrackManager instance
        frame_idx: Mevcut frame numarası
        fps: Anlık FPS değeri
        qod_count: QoD tetiklenen araç sayısı

    Returns:
        Görselleştirilmiş frame (orijinal değiştirilmez)
    """
    vis = frame.copy()
    active_tracks = tracker.get_active_tracks()
    alerts = []

    # ── Her araç için çizim ──
    for track in active_tracks:
        if not track.bbox_history:
            continue

        bbox = track.bbox_history[-1]
        is_critical = track.criticality_score > CRITICALITY_THRESHOLD

        # ── Araç kutusu ──
        color = COLOR_VEHICLE_CRITICAL if is_critical else COLOR_VEHICLE_NORMAL
        thickness = VIS_THICKNESS_CRITICAL if is_critical else VIS_THICKNESS_NORMAL

        cv2.rectangle(
            vis,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color,
            thickness,
        )

        # ── Araç etiketi ──
        label = (
            f"{track.label} | "
            f"ID:{track.track_id[:8]} | "
            f"{track.detection_conf:.0%}"
        )
        _draw_label(vis, label, bbox[0], bbox[1] - 10, color)

        # ── Kabin nesneleri (bbox çizimi) ──
        for obj in track.cabin_objects:
            obj_bbox = obj["bbox"]
            cv2.rectangle(
                vis,
                (obj_bbox[0], obj_bbox[1]),
                (obj_bbox[2], obj_bbox[3]),
                COLOR_CABIN_OBJECT,
                2,
            )
            obj_label = f"{obj['label']} {obj['confidence']:.0%}"
            _draw_label(
                vis, obj_label,
                obj_bbox[0], obj_bbox[1] - 5,
                COLOR_CABIN_OBJECT,
            )

        # ── Plaka bbox — sadece okuma tamamlanmadıysa göster ──
        plate_read = track.plate_text != "" and track.plate_confidence >= 0.75
        if track.plate_bbox_abs and not plate_read:
            pb = track.plate_bbox_abs

            if track.plate_corners_abs is not None:
                cv2.polylines(
                    vis,
                    [track.plate_corners_abs],
                    isClosed=True,
                    color=COLOR_PLATE_BBOX,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            else:
                cv2.rectangle(
                    vis,
                    (pb[0], pb[1]),
                    (pb[2], pb[3]),
                    COLOR_PLATE_BBOX,
                    2,
                )
            _draw_label(
                vis, "PLAKA: [okuma bekleniyor]",
                pb[0], pb[3] + 15,
                COLOR_PLATE_BBOX,
            )

        # ── Durum Paneli (araç kutusunun sağ üstü) ──
        violations = []
        if track.phone_detected:
            violations.append(("TELEFON", track.phone_score, (55, 55, 240)))
            alerts.append(f"TELEFON: Arac {track.track_id[:8]}")
        if getattr(track, "cigarette_detected", False):
            violations.append(("SIGARA", track.cigarette_score, (0, 140, 255)))
            alerts.append(f"SIGARA: Arac {track.track_id[:8]}")
        if getattr(track, "drowsy_detected", False):
            violations.append(("UYUKLAMA", track.perclos, (0, 0, 255)))
            alerts.append(f"UYUKLAMA: Arac {track.track_id[:8]}")
        if getattr(track, "yawn_detected", False) and not getattr(track, "drowsy_detected", False):
            violations.append(("ESNEME", track.mar, (0, 200, 255)))
        # TODO: emniyet kemeri eklendiğinde buraya eklenir
        # if getattr(track, "no_seatbelt", False):
        #     violations.append(("KEMER YOK", 1.0, (0, 200, 255)))

        # FaceMesh metriklerini topla
        face_data = {
            "ear_l": getattr(track, "ear_left", 0.0),
            "ear_r": getattr(track, "ear_right", 0.0),
            "perclos": getattr(track, "perclos", 0.0),
            "mar": getattr(track, "mar", 0.0),
        }

        _draw_status_panel(
            vis, bbox, track.criticality_score, violations,
            is_critical, qod_count,
            track.plate_text if plate_read else "",
            face_data,
        )

    # ── HUD paneli ──
    _draw_hud(vis, frame_idx, fps, len(active_tracks), qod_count, alerts, active_tracks)

    return vis


def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple,
    bold: bool = False,
) -> None:
    """
    Arka planda yarı saydam kutu ile etiket yazar.
    Okunabilirliği artırır — karmaşık arka planlarda bile.
    """
    font_scale = VIS_FONT_SCALE
    thickness = 2 if bold else 1

    (tw, th), baseline = cv2.getTextSize(
        text, _FONT, font_scale, thickness
    )

    # Sınır kontrolü — label frame dışına taşmasın
    y = max(th + 4, y)
    x = max(0, x)

    # Arka plan kutusu
    cv2.rectangle(
        img,
        (x, y - th - 4),
        (x + tw + 4, y + baseline),
        (0, 0, 0),
        cv2.FILLED,
    )

    # Metin
    cv2.putText(
        img, text,
        (x + 2, y - 2),
        _FONT,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_status_panel(
    img: np.ndarray,
    bbox: list,
    risk_score: float,
    violations: list,
    is_critical: bool,
    qod_count: int,
    plate_text: str,
    face_data: dict = None,
) -> None:
    """
    Araç kutusunun sağ üstüne kompakt durum paneli çizer.

    Gösterir:
        - Sürücü Profili: TEHLIKELI / GUVENLI
        - QoD durumu (aktif/pasif)
        - İhlal nedenleri (TELEFON, SIGARA vb.)
        - Plaka metni (okunduysa)

    Args:
        img: Çizim yapılacak frame
        bbox: [x1, y1, x2, y2] araç bbox
        risk_score: 0-100 arası kritiklik skoru
        violations: [(isim, skor, renk_bgr), ...]
        is_critical: QoD tetiklendi mi
        qod_count: Aktif QoD oturum sayısı
        plate_text: Okunan plaka metni ("" = henüz okunmadı)
    """
    FS = 0.38
    FS_L = 0.42
    ROW_H = 18
    PAD = 6

    C_GRAY = (150, 150, 150)
    C_GREEN = (50, 210, 50)
    C_RED = (55, 55, 240)
    C_CYAN = (200, 200, 0)
    C_PLATE = (220, 80, 220)

    is_dangerous = len(violations) > 0

    # ── Panel içerik satırları hesapla ──
    rows = []  # (text, color, bold)

    # Plaka — aracın kimliği, en üstte
    if plate_text:
        rows.append((plate_text, C_PLATE, True))
        rows.append(None)  # Ayırıcı

    # Sürücü profili
    if is_dangerous:
        rows.append(("SURUCU: TEHLIKELI", C_RED, True))
    else:
        rows.append(("SURUCU: GUVENLI", C_GREEN, True))

    # QoD durumu
    if is_critical and qod_count > 0:
        rows.append(("QoD: AKTIF", C_CYAN, False))
    else:
        rows.append(("QoD: PASIF", C_GRAY, False))

    # Ayırıcı
    rows.append(None)

    # İhlaller
    if violations:
        for v_name, v_score, v_color in violations:
            rows.append((f"  {v_name} {v_score:.0%}", v_color, False))
    else:
        rows.append(("  IHLAL YOK", C_GREEN, False))

    # FaceMesh metrikleri — her zaman göster (veri varsa)
    if face_data:
        ear_l = face_data.get("ear_l", 0.0)
        ear_r = face_data.get("ear_r", 0.0)
        perclos = face_data.get("perclos", 0.0)
        mar = face_data.get("mar", 0.0)
        has_face = (ear_l > 0 or ear_r > 0 or perclos > 0 or mar > 0)
        if has_face:
            rows.append(None)  # Ayirici
            C_EAR = (180, 180, 0)     # Cyan-ish
            C_PERCLOS = (55, 55, 240) if perclos >= 0.40 else (150, 150, 150)
            C_MAR = (0, 200, 255) if mar >= 0.55 else (150, 150, 150)
            ear_avg = (ear_l + ear_r) / 2.0
            rows.append((f"  EAR: {ear_avg:.2f}", C_EAR, False))
            rows.append((f"  PERCLOS: {perclos:.0%}", C_PERCLOS, False))
            rows.append((f"  MAR: {mar:.2f}", C_MAR, False))

    # ── Panel boyutları ──
    panel_w = 155
    for r in rows:
        if r is None:
            continue
        txt, _, bold = r
        th = 1 if not bold else 2
        (tw, _), _ = cv2.getTextSize(txt, _FONT, FS_L if bold else FS, th)
        panel_w = max(panel_w, tw + PAD * 2 + 4)

    sep_count = sum(1 for r in rows if r is None)
    content_count = len(rows) - sep_count
    panel_h = PAD + content_count * ROW_H + sep_count * 8 + PAD

    # ── Panel konumu: bbox sağ üst ──
    px = bbox[2] + 4
    py = bbox[1]

    img_h, img_w = img.shape[:2]
    if px + panel_w > img_w - 4:
        px = bbox[0] - panel_w - 4
    if px < 0:
        px = bbox[0]
    if py + panel_h > img_h:
        py = img_h - panel_h - 4

    # ── Arka plan (ROI-only alpha blend — tam frame copy'den ~30x ucuz) ──
    roi = img[py:py + panel_h, px:px + panel_w]
    dark = np.full_like(roi, (18, 18, 18), dtype=np.uint8)
    cv2.addWeighted(dark, 0.85, roi, 0.15, 0, roi)
    img[py:py + panel_h, px:px + panel_w] = roi

    border_color = C_RED if is_dangerous else (65, 65, 65)
    cv2.rectangle(img, (px, py), (px + panel_w, py + panel_h),
                  border_color, 1)

    # Tehlikeli ise üst kenarda kırmızı accent bar
    if is_dangerous:
        cv2.rectangle(img, (px + 1, py + 1), (px + panel_w - 1, py + 3),
                      C_RED, cv2.FILLED)

    cy = py + PAD

    # ── İçerik çizimi ──
    for r in rows:
        if r is None:
            # Ayırıcı çizgi
            cv2.line(img, (px + PAD, cy + 3), (px + panel_w - PAD, cy + 3),
                     (50, 50, 50), 1)
            cy += 8
            continue

        txt, color, bold = r
        fs = FS_L if bold else FS
        th = 2 if bold else 1

        cv2.putText(img, txt,
                    (px + PAD, cy + ROW_H - 5),
                    _FONT, fs, color, th, cv2.LINE_AA)
        cy += ROW_H


def _draw_hud(
    img: np.ndarray,
    frame_idx: int,
    fps: float,
    vehicle_count: int,
    qod_count: int,
    alerts: List[str],
    active_tracks: list,
) -> None:
    """
    Sol üst köşede profesyonel yarı saydam HUD paneli çizer.

    Bölümler:
        Header  : TEKNOFEST 2026 + Road Safety
        Stats   : Frame / FPS / Araç / QoD
        Risk    : Araç sayısına göre otomatik renk
        Plakalar: Okunan plakalar listesi
    """
    # ── Renk Paleti (BGR) ──
    C_BG     = (18, 18, 18)
    C_HDR_BG = (28, 28, 28)
    C_BORDER = (65, 65, 65)
    C_SEP    = (50, 50, 50)
    C_WHITE  = (235, 235, 235)
    C_GRAY   = (150, 150, 150)
    C_ACCENT = (0, 200, 255)      # Cyan
    C_GREEN  = (50, 210, 50)
    C_YELLOW = (0, 210, 255)      # BGR sarı
    C_RED    = (55, 55, 240)
    C_PLATE  = (220, 80, 220)     # Mor

    PANEL_W = 285
    PAD_X   = 10
    ROW_H   = 22
    HDR_H   = 28
    SPAD    = 5
    FS      = 0.42
    FS_S    = 0.37

    # ── Risk Seviyesi ──
    if vehicle_count <= 1:
        risk_label, risk_color = "DUSUK", C_GREEN
    elif vehicle_count <= 3:
        risk_label, risk_color = "ORTA", C_YELLOW
    else:
        risk_label, risk_color = "YUKSEK", C_RED

    # ── Plaka Listesi ──
    plates = []
    for tr in active_tracks:
        if tr.plate_text:
            plates.append((tr.plate_text, f"{tr.plate_confidence:.0%}", True))
        elif tr.plate_bbox_abs:
            plates.append(("[Bekleniyor]", "", False))

    n_plate_rows = max(len(plates), 1)

    # ── Panel Yüksekliği ──
    panel_h = (
        HDR_H
        + 1 + SPAD * 2          # sep
        + ROW_H * 2             # Frame/FPS + Araç/QoD
        + SPAD + 1 + SPAD * 2   # sep
        + ROW_H                 # Risk
        + SPAD + 1 + SPAD * 2   # sep
        + ROW_H                 # "PLAKALAR" başlık
        + ROW_H * n_plate_rows  # plaka satırları
        + SPAD                  # alt boşluk
    )

    x0, y0 = 8, 8
    x1 = x0 + PANEL_W
    y1 = y0 + panel_h

    # ── Alpha Blended Arka Plan (ROI-only — tam frame copy'den ~30x ucuz) ──
    hud_roi = img[y0:y1, x0:x1]
    dark_hud = np.full_like(hud_roi, C_BG, dtype=np.uint8)
    cv2.addWeighted(dark_hud, 0.82, hud_roi, 0.18, 0, hud_roi)
    img[y0:y1, x0:x1] = hud_roi
    cv2.rectangle(img, (x0, y0), (x1, y1), C_BORDER, 1)

    # ── Header ──
    hdr_y1 = y0 + HDR_H
    hdr_h = hdr_y1 - (y0 + 1)
    hdr_w = (x1 - 1) - (x0 + 1)
    if hdr_h > 0 and hdr_w > 0:
        hdr_roi = img[y0 + 1:hdr_y1, x0 + 1:x1 - 1]
        dark_hdr = np.full_like(hdr_roi, C_HDR_BG, dtype=np.uint8)
        cv2.addWeighted(dark_hdr, 0.7, hdr_roi, 0.3, 0, hdr_roi)
        img[y0 + 1:hdr_y1, x0 + 1:x1 - 1] = hdr_roi
    cv2.rectangle(img, (x0 + 1, y0 + 1), (x0 + 4, hdr_y1), C_ACCENT, cv2.FILLED)
    cv2.putText(img, "TEKNOFEST 2026",
                (x0 + PAD_X + 4, y0 + 19),
                _FONT, 0.48, C_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(img, "Road Safety",
                (x0 + 172, y0 + 19),
                _FONT, 0.33, C_GRAY, 1, cv2.LINE_AA)

    cy = hdr_y1

    def _sep() -> None:
        nonlocal cy
        cv2.line(img, (x0 + PAD_X, cy + SPAD),
                 (x1 - PAD_X, cy + SPAD), C_SEP, 1)
        cy += 1 + SPAD * 2

    def _stat_row(lbl_l: str, val_l: str,
                  lbl_r: str, val_r: str,
                  vc: tuple = C_WHITE) -> None:
        nonlocal cy
        mid = x0 + PANEL_W // 2
        ty  = cy + ROW_H - 6
        cv2.putText(img, lbl_l, (x0 + PAD_X, ty),
                    _FONT, FS_S, C_GRAY, 1, cv2.LINE_AA)
        cv2.putText(img, val_l, (x0 + PAD_X + 52, ty),
                    _FONT, FS, C_WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, lbl_r, (mid, ty),
                    _FONT, FS_S, C_GRAY, 1, cv2.LINE_AA)
        cv2.putText(img, val_r, (mid + 45, ty),
                    _FONT, FS, vc, 1, cv2.LINE_AA)
        cy += ROW_H

    _sep()
    _stat_row("Frame :", str(frame_idx), "FPS   :", f"{fps:.1f}")
    _stat_row("Arac  :", str(vehicle_count), "QoD   :", str(qod_count),
              vc=C_YELLOW if qod_count > 0 else C_WHITE)

    _sep()

    # ── Risk Satırı ──
    sq = 9
    sq_y = cy + (ROW_H - sq) // 2
    cv2.rectangle(img, (x0 + PAD_X, sq_y),
                  (x0 + PAD_X + sq, sq_y + sq), risk_color, cv2.FILLED)
    cv2.putText(img, "RISK :",
                (x0 + PAD_X + sq + 6, cy + ROW_H - 6),
                _FONT, FS_S, C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(img, risk_label,
                (x0 + PAD_X + sq + 52, cy + ROW_H - 6),
                _FONT, FS, risk_color, 1, cv2.LINE_AA)
    cy += ROW_H

    _sep()

    # ── Plaka Bölümü ──
    cv2.putText(img, "PLAKALAR",
                (x0 + PAD_X, cy + ROW_H - 6),
                _FONT, FS_S, C_GRAY, 1, cv2.LINE_AA)
    cy += ROW_H

    if plates:
        for plate_txt, plate_conf, confirmed in plates[:3]:
            sq_y = cy + (ROW_H - 7) // 2
            p_color = C_PLATE if confirmed else C_GRAY
            cv2.rectangle(img, (x0 + PAD_X, sq_y),
                          (x0 + PAD_X + 7, sq_y + 7), p_color, cv2.FILLED)
            display = f"{plate_txt}  {plate_conf}" if plate_conf else plate_txt
            cv2.putText(img, display,
                        (x0 + PAD_X + 13, cy + ROW_H - 6),
                        _FONT, FS, p_color, 1, cv2.LINE_AA)
            cy += ROW_H
    else:
        cv2.putText(img, "Plaka Yok",
                    (x0 + PAD_X + 13, cy + ROW_H - 6),
                    _FONT, FS, C_GRAY, 1, cv2.LINE_AA)
