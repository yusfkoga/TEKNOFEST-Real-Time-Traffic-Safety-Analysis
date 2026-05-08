"""
TEKNOFEST 2026 — Pipeline Konfigürasyon Dosyası.
Tüm sabitler burada merkezi olarak tanımlanır.
Hiçbir modül kendi sabitini hardcode etmez.
"""

# ══════════════════════════════════════════════════════
# Model Yolları
# ══════════════════════════════════════════════════════

TIER1_MODEL_PATH = "yolo11s.pt"              # Tier-1: Araç tespiti — hız öncelikli (SMALL)
TIER2A_MODEL_PATH = "yolo11s.pt"             # Tier-2a: Kabin analizi — 11s + SAHI (rapor kararı: aynı model, fark QoD+SAHI)
TIER2B_MODEL_PATH = "models/plate_yolo11s_turkiye.pt"  # Tier-2b: Plaka YOLO 11s fine-tuned + SAHI
# Fine-tuned model yoksa HuggingFace fallback:
PLATE_HF_REPO = "morsetechlab/yolov11-license-plate-detection"
PLATE_HF_FILENAME = "license-plate-finetune-v1l.pt"  # HF repo dosya adı (mevcut en iyi versiyon)

# ══════════════════════════════════════════════════════
# Tier-1: Araç Tespiti Parametreleri
# ══════════════════════════════════════════════════════

TIER1_CLASSES = [2, 3, 5, 7]     # COCO: car, motorcycle, bus, truck
TIER1_CONF = 0.40                # Minimum güven eşiği — dengelenmiş hassasiyet
TIER1_IMGSZ = 640                # GPU var → 640 kullan (doğruluk öncelikli)
# NOT: DETECTION_EVERY_N kaldırıldı — BoTSORT Kalman filtresi için her frame gerekli

# ══════════════════════════════════════════════════════
# Tier-2a: Kabin Analizi Parametreleri
# ══════════════════════════════════════════════════════

TIER2A_CLASSES = [0]             # COCO: person — telefon tespiti MediaPipe Pose'a devredildi
TIER2A_CONF = 0.40               # Logo/yansıma false positive'lerini keser, cam arkası kişi ~0.35-0.70 arası
TIER2A_IMGSZ = 320               # 640 → 320: Telefon tespiti artık MediaPipe'a devredildi, kişi tespiti 320'de yeterli
CABIN_EVERY_N = 4                # Her 4 frame'de kabin analizi — kısa süreli ihlalleri kaçırmamak için
CABIN_MIN_CROP_PX = 150          # Crop'un en/boy'u bundan küçükse analiz atlanır (bulanık büyütme güvenilmez)

# ══════════════════════════════════════════════════════
# SAHI (Slicing Aided Hyper Inference) Parametreleri
# ══════════════════════════════════════════════════════
# Rapor kararı: SAHI tüm frame'e DEĞİL, sadece kabin/plaka crop'una uygulanır.
# Kabin crop'u 3-4 örtüşen tile'a bölünür → küçük nesnelerin efektif boyutu ~4x artar.
# GPU batch inference ile tüm tile'lar tek seferde işlenir.

SAHI_SLICE_WIDTH = 160           # Tile genişliği (px) — 320px crop → ~3-4 tile
SAHI_SLICE_HEIGHT = 160          # Tile yüksekliği (px)
SAHI_OVERLAP_RATIO = 0.25        # %25 örtüşme — kenar nesnelerinin kaybolmasını önler
SAHI_POSTPROCESS_TYPE = "NMS"    # NMS veya NMM — örtüşen tile çıktılarını birleştirme
SAHI_POSTPROCESS_MATCH_THRESHOLD = 0.5  # NMS IoU eşiği — tile-arası duplikasyon filtresi
SAHI_CONF = 0.40                 # SAHI inference conf eşiği (TIER2A_CONF ile aynı)

# ══════════════════════════════════════════════════════
# Tier-2b: Plaka Tespiti Parametreleri
# ══════════════════════════════════════════════════════

TIER2B_CONF = 0.15               # Karanlık plakalara şans vermek için düşürüldü. (Geometrik filtreler yanlış pozitifleri eler)
TIER2B_IMGSZ = 640               # GPU var → plaka tespitinde yüksek çözünürlük
PLATE_EVERY_N = 4                # Her 4 fresh-detection frame'de bir plaka tespiti
                                 # (SAHI tile inference her frame'de ağır — aralıklı çalıştır)

# Plaka arama bölgesi — crop'un üst kısmını atla
# Plakalar araçların alt %55'inde bulunur
PLATE_CROP_TOP_RATIO = 0.45      # Crop'un üst %45'ini atla → sadece alt %55'te ara

# ══════════════════════════════════════════════════════
# Araç Crop Parametreleri
# ══════════════════════════════════════════════════════

CROP_PADDING = 0.05              # %5 padding — bağlam koruması için

# ══════════════════════════════════════════════════════
# BoTSORT Tracker Parametreleri
# ══════════════════════════════════════════════════════
# BoTSORT: Ultralytics built-in (Kalman + ReID + IoU assignment)
# Konfigurasyonu botsort.yaml dosyasında bulunur.
# Aşağıdakiler TrackManager (pipeline state yönetimi) içindir.

MAX_AGE = 30                     # Frame sayısı — bu kadar kaybolursa track silinir
MAX_TRACKS = 50                  # Maksimum eşzamanlı track (LRU eviction)
BBOX_HISTORY_LEN = 5             # Her track için saklanan son bbox sayısı

# ══════════════════════════════════════════════════════
# OCR Parametreleri (fast-plate-ocr)
# ══════════════════════════════════════════════════════

OCR_MODEL_NAME = "cct-s-v2-global-model"  # CCT-S v2 global — plakaya özel hafif ONNX model
OCR_RATE_LIMIT_FRAMES = 15      # Aynı track_id için minimum frame arası
                                 # (OCR boş dönerse hemen tekrar çalışmasın → FPS korunur)

# ══════════════════════════════════════════════════════
# Türk Plaka Fiziksel Sabitleri (CV Fallback)
# ══════════════════════════════════════════════════════
# Gerçek boyut: 520mm × 110mm → oran 4.727:1
# Tolerans aralığı perspektif bozulmalarını kapsar

PLATE_ASPECT_MIN = 2.5           # Minimum en-boy oranı (gevşetildi, perspektif toleransı)
PLATE_ASPECT_MAX = 6.0           # Maksimum en-boy oranı
PLATE_MIN_AREA = 800             # Minimum alan (px²) — çok küçük adayları ele
PLATE_MAX_AREA = 50000           # Maksimum alan (px²) — çok büyük yanlış pozitifleri ele

# Plaka-araç oranı validasyonu (YOLO/HF detektörler için)
PLATE_MIN_WIDTH_RATIO = 0.05     # Plaka genişliği < crop genişliğinin %5'i → reddet
PLATE_MAX_WIDTH_RATIO = 0.45     # Plaka genişliği > crop genişliğinin %45'i → reddet (Tampon ızgarası gibi yanılgıları kesin olarak keser)
PLATE_DET_ASPECT_MIN = 1.5       # Plaka bbox aspect ratio minimum (YOLO/HF)
PLATE_DET_ASPECT_MAX = 7.0       # Plaka bbox aspect ratio maximum (YOLO/HF)

# ══════════════════════════════════════════════════════
# Türk Plaka Format Validasyonu
# ══════════════════════════════════════════════════════

PLATE_REGEX = r"^\d{2}\s?[A-Z]{1,3}\s?\d{2,4}$"
# Format: XX YYY ZZZZ (il kodu + harf grubu + numara)
# Örnekler: 34 ABC 1234, 06 A 0001, 01 AB 123

# ══════════════════════════════════════════════════════
# Kritiklik Skoru Parametreleri
# ══════════════════════════════════════════════════════

CRITICALITY_THRESHOLD = 60       # Bu değerin üstü = kritik araç → QoD tetiklenir (0-100 scale)

# ══════════════════════════════════════════════════════
# Pose Analizi Parametreleri (MediaPipe)
# ══════════════════════════════════════════════════════

POSE_PHONE_THRESHOLD = 0.15      # Bilek ↔ Kulak normalize L2 mesafesi eşiği (0-1 koordinat uzayı)
                                 # 0.15 → 640px crop'ta ~96px — kulağa değen el = telefon
POSE_MIN_VISIBILITY = 0.30       # Bu değerin altındaki landmark'lar atlanır (okluzyona karşı koruma)

# ══════════════════════════════════════════════════════
# Face Mesh Parametreleri (MediaPipe)
# ══════════════════════════════════════════════════════
# EAR: Eye Aspect Ratio — göz kapanma oranı (0=kapalı, ~0.25=açık)
# PERCLOS: Percentage of Eye Closure — son N frame'de göz kapalılık yüzdesı
# MAR: Mouth Aspect Ratio — esneme/çene açıklığı

FACE_MESH_EAR_THRESHOLD = 0.20   # EAR bu değerin altına düşerse göz kapalı sayılır
FACE_MESH_PERCLOS_THRESHOLD = 0.40  # Son N frame'de %40+ göz kapalı = uyuklama
FACE_MESH_PERCLOS_WINDOW = 30    # PERCLOS pencere boyutu (frame) — ~1sn @ 30fps
FACE_MESH_MAR_THRESHOLD = 0.65   # MAR bu değerin üstüne çıkarsa esneme sayılır
FACE_MESH_MIN_DETECTION_CONF = 0.4  # Face Mesh yüz tespit güven eşiği
FACE_MESH_EVERY_N = 2            # Her 2 kabin frame'de bir Face Mesh çalıştır
                                 # (CABIN_EVERY_N * FACE_MESH_EVERY_N = 8 frame aralık)

# ══════════════════════════════════════════════════════
# TensorRT Entegrasyonu
# ══════════════════════════════════════════════════════
# .engine dosyası varsa TensorRT kullanılır.
# Yoksa ilk çalıştırmada otomatik export edilir (1 kere, ~2-5dk).

TENSORRT_ENABLED = True          # TensorRT export/kullanımı aktif mi
TENSORRT_PRECISION = "fp16"      # "fp16" veya "int8" — FP16 daha güvenli, INT8 daha hızlı
                                 # INT8 kalibrasyon verisi gerektirir

# ══════════════════════════════════════════════════════
# Frame Çözünürlük Sınırlaması
# ══════════════════════════════════════════════════════

MAX_FRAME_WIDTH = 1920           # Capture anında bu genişliğe indirgenir (4K+ için 4x bellek tasarrufu)
                                 # YOLO zaten içinde 640px'e resize yapar — 4K'ın etkisi sıfır
                                 # None = resize yok (orn. 1080p kaynaklar)

# ══════════════════════════════════════════════════════
# Queue Boyutları — Bellek Yönetimi
# ══════════════════════════════════════════════════════

CAPTURE_QUEUE_SIZE = 2           # Canlı mod: düşük latency, eski frame'i at
VIDEO_QUEUE_SIZE = 512           # Video modu: büyük buffer, frame kaybı yok
OCR_QUEUE_SIZE = 10              # OCR tampon

# ══════════════════════════════════════════════════════
# Thread Pool Boyutları
# ══════════════════════════════════════════════════════

DETECTION_WORKERS = 1            # Tier-1 — tek thread yeterli (GPU bottleneck)
ANALYSIS_WORKERS = 2             # Tier-2a + 2b — 2 thread paralel analiz
OCR_WORKERS = 1                  # OCR — tek thread (ONNX session thread-safe değil)

# ══════════════════════════════════════════════════════
# Bellek Yönetimi
# ══════════════════════════════════════════════════════

DEAD_TRACK_CLEANUP_INTERVAL = 500  # Her 500 frame'de ölü track temizliği

# ══════════════════════════════════════════════════════
# Görselleştirme Renkleri (BGR formatı — OpenCV standardı)
# ══════════════════════════════════════════════════════

COLOR_VEHICLE_NORMAL = (0, 255, 0)       # Yeşil — normal araç kutusu
COLOR_CABIN_OBJECT = (0, 165, 255)       # Turuncu — kabin nesnesi
COLOR_PLATE_BBOX = (255, 0, 255)         # Mor — plaka bbox
COLOR_VEHICLE_CRITICAL = (0, 0, 255)     # Kırmızı — kritik araç (score > 0.6)

# ══════════════════════════════════════════════════════
# Görselleştirme Parametreleri
# ══════════════════════════════════════════════════════

VIS_FONT_SCALE = 0.40            # Yazı boyutu (1080p frame için — 4K'daki küçük etiket görünümü)
VIS_THICKNESS_NORMAL = 2         # Normal araç kutusu kalınlığı
VIS_THICKNESS_CRITICAL = 3       # Kritik araç kutusu kalınlığı (belirgin)

# ══════════════════════════════════════════════════════
# COCO Sınıf Etiketleri (Kullanılan alt küme)
# ══════════════════════════════════════════════════════

COCO_LABELS = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    67: "cell phone",
}
