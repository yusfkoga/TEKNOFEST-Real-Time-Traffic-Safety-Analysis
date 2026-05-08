"""TensorRT Engine Compile — Teknik Rapor PDF Oluşturucu."""
from fpdf import FPDF

class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "TEKNOFEST 2026 - TensorRT Teknik Rapor", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Sayfa {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(25, 60, 120)
        self.ln(4)
        self.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(25, 60, 120)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(3)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.ln(2)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        x = self.get_x()
        self.cell(10, 5.5, "  - " + text, new_x="LMARGIN", new_y="NEXT")

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(25, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(240, 245, 255)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6.5, str(val), border=1, fill=True, align="C")
            self.ln()
            fill = not fill
        self.ln(2)


def build_report():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Baslik ──
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 50, 100)
    pdf.cell(0, 15, "TensorRT Engine Compile", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "YOLO Model Optimizasyonu - Teknik Rapor", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_draw_color(25, 60, 120)
    pdf.set_line_width(0.5)
    pdf.line(50, pdf.get_y(), 160, pdf.get_y())
    pdf.ln(6)

    # Proje bilgisi
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, "Proje: TEKNOFEST 2026 - 5G & Yapay Zeka ile Akilli Yol Guvenligi", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "GPU: NVIDIA GeForce RTX 4050 Laptop  |  CUDA 12.4  |  TensorRT 10.16.1", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # ══════════════════════════════════════════
    pdf.section_title("1. Ne Yapiliyor?")
    pdf.body(
        "TensorRT, YOLO modelini PyTorch formatindan (.pt) GPU'ya ozel optimize edilmis "
        "binary formata (.engine) donusturuyor. Bu islem 3 asamada gerceklesir:"
    )

    pdf.sub_title("Asama 1 - Model Cikarimi (.pt -> .onnx)")
    pdf.body(
        "PyTorch modelinin katman yapisi (convolution, batch norm, activation vb.) "
        "platform-bagimsiz ONNX formatina aktarilir. Bu adim hizlidir (~5-10sn)."
    )

    pdf.sub_title("Asama 2 - Katman Fuzyonu")
    pdf.body(
        "Ardisik katmanlar birlestirilir. Ornegin Conv2d + BatchNorm + ReLU normalde "
        "3 ayri GPU cagrisidir. TensorRT bunlari tek bir CUDA kernel'ina birlestirir. "
        "Bellek transferi azalir, hiz artar."
    )

    pdf.sub_title("Asama 3 - Kernel Autotuning (en uzun suren)")
    pdf.body(
        "Her birlesmis katman icin GPU'da yuzlerce farkli CUDA kernel implementasyonu "
        "denenir. Her birinin suresi olculur ve bu GPU'ya ozel en hizlisi secilir. "
        "RTX 4050'nin CUDA cekirdek sayisi, bellek bant genisligi ve cache yapisina "
        "gore optimize edilir. Bu yuzden compile edilen .engine dosyasi baska bir "
        "GPU'da calismaz."
    )

    # ══════════════════════════════════════════
    pdf.section_title("2. Canli Akista Avantajlar")

    pdf.sub_title("2.1 Inference Hizi (Ana Kazanim)")
    pdf.add_table(
        ["Model", "PyTorch FP32", "PyTorch FP16", "TensorRT FP16"],
        [
            ["Tier-1 (640px)", "~15-20ms", "~10-15ms", "~2-4ms"],
            ["Tier-2a (320px)", "~8-12ms", "~6-8ms", "~1-2ms"],
        ],
        [48, 48, 48, 48],
    )
    pdf.body(
        "Canli akista her milisaniye onemlidir. 4ms kazanim, 30 FPS'te toplam "
        "%12 frame butcesi demektir."
    )

    pdf.sub_title("2.2 Bellek Verimliligi")
    pdf.body(
        "TensorRT engine, model agirliklarini FP16'ya donusturup sabit bellek havuzu "
        "kullanir. PyTorch'un dinamik bellek tahsisi ve garbage collection overhead'i "
        "yoktur. GPU bellek kullanimi ~%30-40 azalir."
    )

    pdf.sub_title("2.3 Deterministik Gecikme (Jitter Azalmasi)")
    pdf.body(
        "PyTorch'ta inference suresi frame'den frame'e degisebilir (Python GIL, garbage "
        "collection, CUDA stream senkronizasyonu). TensorRT engine compile edilmis C++ "
        "binary oldugundan gecikme neredeyse sabit kalir. Canli akista bu, frame drop "
        "riskini azaltir."
    )

    pdf.sub_title("2.4 CPU Yuku Azalmasi")
    pdf.body(
        "PyTorch inference'i Python katmanindan gecer (operator dispatch, autograd context, "
        "tensor metadata). TensorRT dogrudan GPU'da calisir, CPU sadece giris/cikis "
        "kopyalamasi yapar. CPU'nun asenkron pipeline gorevleri (tracking, OCR, "
        "gorsellestirme) icin daha fazla kapasitesi kalir."
    )

    # ══════════════════════════════════════════
    pdf.section_title("3. Dezavantajlar")

    pdf.sub_title("3.1 Ilk Compile Suresi")
    pdf.body(
        "Her model+boyut kombinasyonu icin bir kez 3-10 dakika beklenir. "
        "Pipeline'da 2 engine vardir:"
    )
    pdf.bullet("yolo11s_640.engine (Tier-1) -> ~5-10dk")
    pdf.bullet("yolo11s_320.engine (Tier-2a) -> ~3-5dk")
    pdf.ln(1)
    pdf.body(
        "Bu sadece ilk calistirmada olur. Sonrasinda .engine dosyasi diskten "
        "~1-2 saniyede yuklenir."
    )

    pdf.sub_title("3.2 GPU'ya Ozel - Tasinabilirlik Yok")
    pdf.body(
        "Compile edilen engine sadece bu GPU'da calisir. Farkli bir bilgisayara veya "
        "GPU'ya tasima durumunda yeniden compile gerekir. Yarisma ortaminda farkli GPU "
        "varsa ilk calistirmada bekleme olacaktir. Ancak fallback mekanizmasi bunu "
        "otomatik halleder."
    )

    pdf.sub_title("3.3 Dinamik Boyut Destegi Yok")
    pdf.body(
        "TensorRT engine sabit giris boyutuna compile edilir (640x640 veya 320x320). "
        "Farkli bir boyut gelirse engine kullanilamaz. Bu yuzden her imgsz icin ayri "
        "engine olusturulur. Pipeline'da bu zaten yonetilmektedir "
        "(yolo11s_640.engine vs yolo11s_320.engine)."
    )

    pdf.sub_title("3.4 Model Guncelleme Maliyeti")
    pdf.body(
        "Model agirliklari degisirse (fine-tuning, yeni versiyon) engine yeniden "
        "compile edilmelidir. .pt dosyasi degisip .engine eski kalirsa yanlis sonuclar "
        "uretir. Bunu onlemek icin engine dosya isimlerinde model adi ve boyut kodlanmistir."
    )

    # ══════════════════════════════════════════
    pdf.section_title("4. Fallback Guvenligi")
    pdf.body(
        "Engine compile basarisiz olursa veya .engine dosyasi yoksa pipeline otomatik "
        "olarak PyTorch FP16'ya duser. Sistem hicbir kosulda durmaz:"
    )
    pdf.ln(1)
    pdf.set_font("Courier", "", 9)
    pdf.set_text_color(40, 40, 40)
    fallback_text = (
        ".engine varsa   ->  TensorRT (en hizli)\n"
        ".engine yoksa   ->  compile dene\n"
        "                      -> basariliysa TensorRT\n"
        "                      -> basarisizsa PyTorch FP16"
    )
    pdf.multi_cell(0, 5, fallback_text)
    pdf.ln(3)

    # ══════════════════════════════════════════
    pdf.section_title("5. Sonuc Karsilastirmasi")
    pdf.add_table(
        ["Metrik", "Eski (PyTorch)", "Yeni (TensorRT)"],
        [
            ["Tier-1 inference", "~12ms", "~3ms"],
            ["Tier-2a inference", "~8ms", "~1.5ms"],
            ["Toplam pipeline FPS", "~6 FPS", "~25-50 FPS (beklenti)"],
            ["Ilk acilis", "Aninda", "+5-10dk (tek sefer)"],
            ["Sonraki acilislar", "Aninda", "Aninda (~1-2sn)"],
            ["GPU bellek", "~1.5GB", "~1GB"],
        ],
        [64, 64, 64],
    )

    pdf.body(
        "Canli akis icin TensorRT net kazanimdir. Tek bedeli ilk seferlik compile "
        "suresi ve GPU'ya bagimlilik - ikisi de kabul edilebilir trade-off'lardir."
    )

    # ══════════════════════════════════════════
    out = "TensorRT_Engine_Compile_Raporu.pdf"
    pdf.output(out)
    print(f"PDF olusturuldu: {out}")


if __name__ == "__main__":
    build_report()
