# TEKNOFEST 2026 — Real-Time Traffic Safety Analysis with 5G QoD Support (Edge AI) 
(template version for preliminary design stage)

# Proje özeti : 
Bu proje; sabit bir yol kenarı kamerasından alınan canlı video akışını yapay zekâ tabanlı algoritmalarla gerçek zamanlı analiz eden, Turkcell Open Gateway platformunun Number Verification ve Quality on Demand (QoD) API’leri üzerinden dinamik ağ kalite yönetimi yapan uçtan uça bir akıllı yol güvenliği sistemidir. Sistem iki katmanlı bir YZ mimarisi üzerine inşa edilmiştir: Birinci katmanda YOLO 11s ile araç tespiti, MediaPipe Pose ile sürücü davranış ön taraması (telefon, sigara) ve MediaPipe Face Mesh ile yorgunluk takibi (EAR/PERCLOS) her frame’de sürekli çalışır. İkinci katmanda, davranış şüphesi oluştuğunda QoD API kademeli olarak tetiklenir ve bant genişliği artırılarak efektif piksel kalitesi yükseltilir. Bu iyileştirilmiş frame’ler üzerinde aynı YOLO 11s modeli + SAHI (Slicing Aided Hyper Inference) ile yalnızca kabin crop’u dilimlenerek detaylı analiz gerçekleştirilir — tüm frame’i çok sayıda kareye bölmek yerine yalnızca ilgili bölge hedeflenir. Bu yaklaşımda doğruluk artışının temel kaynağı daha büyük bir model değil, QoD’un sağladığı temiz görüntü ve SAHI’nin sağladığı efektif çözünürlük artışıdır.
Plaka lokalizasyonu fine-tuned YOLO 11s + SAHI, karakter tanıma ise fast-plate-ocr ile gerçekleştirilmektedir. QoD tetikleme, ikili bir anahtar yerine güven skoruna ters orantılı kademeli bir mekanizma ile yönetilir: en belirsiz tespitler en yüksek ağ desteğini alır. Eğitim verisinin zenginleştirilmesi için mevcut seed veriler üzerinden sentetik veri üretimi planlanmaktadır; bu sayede modelin farklı kamera açılarına, hava koşullarına ve çözünürlük değişimlerine dayanıklılığı artırılır. Projenin temel özgün katkıları; her araç için ayrı sürdürülen sliding-window profilleme yöntemi, Türk Trafik Kanunu’na dayandırılan nesnel risk derecelendirme sistemi ve F1-confidence eğrisinden türetilen veri güdümlü QoD eşik kalibrasyonudur.

# Sistem Ön Tasarımı : 
QoD API Entegrasyonu ve Veri Akışı
(1) Risk skoru 50 eşiğini aştığında FastAPI, Open Gateway /qod/v0/sessions endpoint'ine OAuth 2.0 ile POST isteği atar (Gövde: ueId, asId, qosProfile, duration). 
(2-3) CAMARA API bildirimiyle PCF, SMF üzerinden UPF'yi güncelleyerek ek bant genişliği sağlar.
(4) Döndürülen sessionId FastAPI tarafından saklanır; risk yaklaşık %50’nin altına indiğinde ve %85’in üstüne çıktığında DELETE isteğiyle oturum kapatılıp ağ kaynağı serbest bırakılır. Uygulama katmanında tamamen şeffaf olan bu süreçte geliştirici yalnızca CAMARA REST arayüzünü kullanır.
Kademeli QoD Tetikleme Mantığı Şebeke kaynaklarını verimli kullanmak amacıyla üç kademeli QoD profili uygulanır:
- 0,50–0,65 Arası (Belirsiz Tespit): HIGH profiliyle maksimum bant genişliği talep edilir.
- 0,65–0,80 Arası (Olası Tespit): MEDIUM profiliyle orta düzey destek sağlanır.
- 0,80 Üzeri (Net Tespit): Ek ağ kaynağına gerek duyulmaz ve QoD kapatılır. En belirsiz tespitler en yüksek ağ desteğini alacak şekilde optimize edilmiştir.

Katman 1 — Sürekli Algılama (Her Frame)
- Dış Nesne Tespiti (Tier-1): Mevcut aşamada YOLO 11s ile tüm frame işlenmektedir. NMS-free/anchor-free yapısıyla edge cihazlarda ~2,5 ms sürede çalışır.Model seçimi kesin bir karar olmayıp geliştirme sürecinde elde edilecek benchmark sonuçlarına göre YOLO 11s, 11m veya 11l arasında optimize edilebilir; bu aşamada hız–doğruluk dengesi açısından 11s uygun görülmüştür.
- Sürücü Davranış Ön Taraması: Kabin crop'unda çift MediaPipe paralel çalışır. Pose modeliyle telefon (bilek-kulak) ve sigara (bilek-ağız); Face Mesh ile esneme (MAR) ve yorgunluk (EAR) tespiti yapılır. Son 60 frame'lik PERCLOS > 0,15 ise yorgunluk kabul edilir.
- Hız Tahmini: Yüksek hız, risk skorunu artırıp QoD tetikleme eşiklerini düşürür.
- Emniyet Kemeri ve Plaka: Kabin crop'unda fine-tuned YOLO 11s ile kemer taranır (şüpheliyse Tier-2'ye aktarılır). Araç crop'unda fast-plate-ocr ile Türk plakası okunur. YOLO +SAHI ile birlikte plaka lokalizasyonu uygulanır.
- Ek İhlaller: Kulaklık kullanımı direksiyon bırakma ,ani şerit/çizgi ihlali, aşırı yolcu ve açık kapı/bagaj ;risk skoru formülüne ağırlıklı eklenir.

Katman 2 — Derin Doğrulama (QoD Tetiklendiğinde) QoD ile artan bant genişliği sayesinde aynı YOLO 11s modeli + SAHI çalıştırılır. Geleneksel tam frame SAHI (~250 ms maliyet) yerine, sadece kabin/plaka bölgeleri 3–4 örtüşen tile’a bölünerek küçük nesneler efektif olarak ~4× büyütülür. GPU batch inference ile ek gecikme yalnızca ~4 ms’dir. Tier-2’de kullanılan model, Tier-1 ile aynı ağırlıklara sahiptir; doğruluk kazancı modelden değil QoD’un sağladığı görüntü kalitesi ve SAHI’nin efektif çözünürlük artışından kaynaklanır. Nihai model varyantı geliştirme sürecindeki karşılaştırmalı testlere göre belirlenecektir.

Yorgunluk Doğrulaması: SAHI ile yüz bbox'u tespit edilip ölçeklenerek Face Mesh'e verilir, EAR ve PERCLOS hassasiyeti belirgin ölçüde artar.
Sigara/Telefon Doğrulaması: Tier-1'de tespit edilen şüpheli bölge, QoD ile gelen yüksek kaliteli görüntü üzerinde 3–4 örtüşen tile'lara bölünür. Sigara ve telefon tespiti için özel eğitilmiş YOLO 11s modeli, SAHI'nin sağladığı ~4× efektif çözünürlük artışıyla bu tile'lar üzerinde çalıştırılır. Küçük ve kısmen gizli nesneler (elde tutulan telefon, dudak kenarındaki sigara) bu sayede çok daha yüksek güvenle tespit edilir. Doğruluk kazancı, modelin ağırlıklarından değil; QoD'un sağladığı görüntü kalitesi ile SAHI'nin odaklanmış tile stratejisinin birlikte çalışmasından kaynaklanır.
