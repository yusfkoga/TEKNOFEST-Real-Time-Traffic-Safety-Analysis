"""
TEKNOFEST 2026 — KTK Ağırlıklı Kritiklik Skoru ve Tier-Risk Hesaplama.

Karayolları Trafik Kanunu (2918 Sayılı Kanun) ceza puanı sistemi baz alınır.
KTK'da 1 yılda 100 puana ulaşan sürücünün ehliyetine el konulur.
Bizim pipeline skoru da 0-100 aralığındadır ve doğrudan KTK puanlarıyla eşleşir.

Tespit edilen ihlaller ve KTK karşılıkları:
  ┌─────────────────────┬──────────────┬─────────────┬──────────────────────┐
  │ İhlal               │ KTK Maddesi  │ Ceza Puanı  │ Para Cezası (2026)   │
  ├─────────────────────┼──────────────┼─────────────┼──────────────────────┤
  │ Telefon kullanımı   │ Md. 73/3     │ 20 puan     │ 2.719 TL             │
  │ Emniyet kemeri yok  │ Md. 78/1-a   │ 15 puan     │ 1.245 TL             │
  │ Uyuklama (PERCLOS)  │ TCK Md. 179  │ 30 puan*    │ Hapis (1-6 yıl)      │
  │ Esneme (MAR)        │ —            │ 10 puan     │ Uyarı                │
  │ Sigara kullanımı    │ —            │ 10 puan     │ Dikkat dağıtıcı      │
  └─────────────────────┴──────────────┴─────────────┴──────────────────────┘
  * TCK Md. 179: "Trafik güvenliğini tehlikeye sokma" — KTK ceza puanı yok
    ancak hapis cezası olan tek ihlal. Sistemde en yüksek ağırlık.

İhlaller kümülatiftir: telefon(20) + uyuklama(30) = 50 puan.
score ≥ 60 → QoD tetiklenir (yüksek bant genişliği).
"""

import logging
from typing import Dict, List, Tuple

from config import (
    FACE_MESH_PERCLOS_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
# KTK Ceza Puanı Ağırlıkları (0-100 ölçeği)
# ══════════════════════════════════════════════════════
# Bu değerler doğrudan KTK/TCK'dan alınmıştır.
# Pipeline'ın risk skoru = KTK ceza puanı toplamı.

KTK_TELEFON = 20            # KTK Md. 73/3 — Seyir halinde telefon kullanma
KTK_EMNIYET_KEMERI = 15     # KTK Md. 78/1-a — Emniyet kemeri takmama
KTK_UYUKLAMA = 30           # TCK Md. 179 — Trafik güvenliğini tehlikeye sokma (hapis)
KTK_ESNEME = 10             # Uyarı seviyesi — uyuklama öncüsü
KTK_SIGARA = 10             # Dikkat dağıtıcı davranış (KTK'da doğrudan tanımsız)

# Ciddi uyuklama (PERCLOS ≥ 0.60) → acil müdahale
PERCLOS_SEVERE_THRESHOLD = 0.60


def compute_tier1_base(track, frame_shape: Tuple[int, ...]) -> float:
    """
    Tier-1 (VehicleDetector) verisine göre temel araç riskini hesaplar.
    Yakınlık bazlı bir skor üretir.
    Tier-2'nin uyanması için (Risk >= 45) aracın ekranda belirgin
    bir yer (%5 - %15) kaplamaya başlaması hedeflenir.

    NOT: Bu skor KTK ihlali DEĞİLDİR — sadece Tier-2 analizi
    tetiklemek için kullanılır. İhlal yoksa risk 0 kalır.
    """
    frame_area = frame_shape[0] * frame_shape[1]

    if not track.bbox_history:
        return 0.0

    x1, y1, x2, y2 = track.bbox_history[-1]
    bbox_area = (x2 - x1) * (y2 - y1)

    ratio = bbox_area / frame_area

    # ratio %15 olduğunda max skor 65 dönsün.
    # Bu sayede Tier-2 (11s + SAHI) uyanabilir.
    score = min(ratio * (65.0 / 0.15), 75.0)

    return score


def compute_risk(tier2a_objs: List[Dict], track, tier1_base: float) -> dict:
    """
    KTK ağırlıklı kural ihlali skoru hesaplar.

    Risk skoru aracın büyüklüğü (proximity) ile ARTMAZ.
    Masum bir sürücü ekranda büyük görünüyor diye
    QoD bant genişliğini sömüremez.

    Sadece tespit edilen ihlallerin KTK ceza puanları toplanır.

    Args:
        tier2a_objs: CabinAnalyzer'dan dönen nesneler
        track: Araç track bilgisi (TrackState)
        tier1_base: Tier-1'den gelen baz risk (sadece Tier-2 uyandırmak için)

    Returns:
        {"score": float, "violations": list, "force_qod_l": bool,
         "qod_profile": str, "ktk_detail": dict}
    """
    score = 0.0
    violations = []       # Tespit edilen ihlal etiketleri
    ktk_detail = {}       # Her ihlal → {"puan": int, "madde": str, "ceza_tl": str}

    # ── Telefon Kullanımı (KTK Md. 73/3) ──
    # Tespit: MediaPipe Pose — bilek ↔ kulak mesafesi (davranış bazlı)
    # NOT: YOLO cell phone (class 67) kaldırıldı — cam arkasından güvenilmez.
    # İleride fine-tuned model gelince YOLO tespiti tekrar eklenebilir.
    phone = getattr(track, "phone_detected", False)

    if phone:
        # Güven ağırlıklı: phone_score 0-1 → KTK puanı oranlanır
        phone_conf = max(getattr(track, "phone_score", 0.5), 0.5)
        puan = KTK_TELEFON * phone_conf
        score += puan
        violations.append("TELEFON")
        ktk_detail["telefon"] = {
            "puan": round(puan, 1),
            "madde": "KTK Md. 73/3",
            "ceza_tl": "2.719 TL",
            "conf": round(phone_conf, 2),
        }

    # ── Sigara Kullanımı ──
    cigarette = getattr(track, "cigarette_detected", False)
    if cigarette:
        cig_conf = max(getattr(track, "cigarette_score", 0.5), 0.5)
        puan = KTK_SIGARA * cig_conf
        score += puan
        violations.append("SIGARA")
        ktk_detail["sigara"] = {
            "puan": round(puan, 1),
            "madde": "Dikkat dağıtıcı",
            "ceza_tl": "—",
            "conf": round(cig_conf, 2),
        }

    # ── Uyuklama — PERCLOS (TCK Md. 179) ──
    perclos = getattr(track, "perclos", 0.0)
    drowsy = getattr(track, "drowsy_detected", False)
    severe_drowsy = (perclos >= PERCLOS_SEVERE_THRESHOLD)

    if drowsy:
        # PERCLOS oranına göre ağırlıklandır:
        # perclos=0.40 → KTK_UYUKLAMA * 0.40/0.40 = 30 puan (tam)
        # perclos=0.80 → 30 puan (max cap)
        perclos_ratio = min(perclos / FACE_MESH_PERCLOS_THRESHOLD, 1.5)
        puan = KTK_UYUKLAMA * perclos_ratio
        score += puan
        violations.append("UYUKLAMA")
        ktk_detail["uyuklama"] = {
            "puan": round(puan, 1),
            "madde": "TCK Md. 179",
            "ceza_tl": "Hapis (1-6 yıl)",
            "perclos": round(perclos, 3),
            "seviye": "KRITIK" if severe_drowsy else "UYARI",
        }

    # ── Esneme — MAR (uyuklama öncüsü) ──
    yawn = getattr(track, "yawn_detected", False)
    if yawn and not drowsy:
        # Esneme tek başına düşük puan — ama uyuklama ile birleşmez (çifte sayma önlemi)
        mar_val = getattr(track, "mar", 0.0)
        puan = KTK_ESNEME * min(mar_val / 0.65, 1.0)
        score += puan
        violations.append("ESNEME")
        ktk_detail["esneme"] = {
            "puan": round(puan, 1),
            "madde": "—",
            "ceza_tl": "Uyarı",
            "mar": round(mar_val, 3),
        }

    # ── Emniyet Kemeri (KTK Md. 78/1-a) ──
    # TODO: Özel model ile aktifleşecek (kabin YOLO seatbelt sınıfı)
    no_seatbelt = getattr(track, "no_seatbelt_detected", False)
    if no_seatbelt:
        puan = KTK_EMNIYET_KEMERI
        score += puan
        violations.append("KEMER_YOK")
        ktk_detail["emniyet_kemeri"] = {
            "puan": puan,
            "madde": "KTK Md. 78/1-a",
            "ceza_tl": "1.245 TL",
        }

    # ── Skor Finalize ──
    score = min(score, 100.0)
    has_violation = len(violations) > 0

    # Ciddi uyuklama → hysteresis bypass (acil müdahale)
    force_qod = severe_drowsy

    return {
        "score": score,
        "violations": violations,
        "ktk_detail": ktk_detail,

        # force_qod_l: Ciddi uyuklama (PERCLOS ≥ 0.60) ise hysteresis atlanır.
        # Diğer ihlaller normal zamanlama ile çalışır (false positive koruması).
        "force_qod_l": force_qod,

        # Bant genişliği profili
        "qod_profile": "THROUGHPUT_L" if has_violation else "THROUGHPUT_S",
    }


def get_trigger_reasons(risk_info: dict) -> List[str]:
    """Risk skoru loglaması için okunaklı nedenler listesi."""
    reasons = []
    violations = risk_info.get("violations", [])

    if risk_info.get("force_qod_l"):
        reasons.append("UYUKLAMA_KRITIK (TCK Md. 179 — ACIL)")

    for v in violations:
        detail = risk_info.get("ktk_detail", {})
        if v == "TELEFON":
            d = detail.get("telefon", {})
            reasons.append(
                f"TELEFON (KTK Md.73/3 | {d.get('puan', 20)} puan | {d.get('ceza_tl', '2.719 TL')})"
            )
        elif v == "SIGARA":
            d = detail.get("sigara", {})
            reasons.append(
                f"SIGARA ({d.get('puan', 10)} puan | dikkat dağıtıcı)"
            )
        elif v == "UYUKLAMA":
            d = detail.get("uyuklama", {})
            reasons.append(
                f"UYUKLAMA (TCK Md.179 | {d.get('puan', 30)} puan | "
                f"PERCLOS={d.get('perclos', 0):.3f} | {d.get('seviye', 'UYARI')})"
            )
        elif v == "ESNEME":
            d = detail.get("esneme", {})
            reasons.append(
                f"ESNEME ({d.get('puan', 10)} puan | MAR={d.get('mar', 0):.3f})"
            )
        elif v == "KEMER_YOK":
            d = detail.get("emniyet_kemeri", {})
            reasons.append(
                f"KEMER_YOK (KTK Md.78/1-a | {d.get('puan', 15)} puan | {d.get('ceza_tl', '1.245 TL')})"
            )

    return reasons
