import asyncio
import json
import logging
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger("qod_manager")

async def mock_camara_create_session(profile: str) -> dict:
    """
    CAMARA uyumlu QoD (Quality on Demand) API Mock.
    Gerçekçi network gecikmesi (RTT) ve standart payload içerir.
    Final yarışmasında gerçek endpoint'e geçmek için sadece bu fonksiyonun 
    içi gerçek bir HTTP isteği ile değiştirilecektir.
    """
    # await asyncio.sleep(0.12)  # Production'da gerçek HTTP RTT olacak
    session_id = f"qod-{uuid4().hex[:8]}"
    payload = {
        "sessionId": session_id,
        "profile": profile,
        "bandwidth": "5Mbps" if profile == "THROUGHPUT_S" else "10Mbps",
        "status": "ACTIVE",
        "timestamp": datetime.utcnow().isoformat()
    }
    logger.info(f"[QoD API] SESSION CREATED | {json.dumps(payload)}")
    return payload

async def mock_camara_delete_session(session_id: str) -> bool:
    """Mock session silme işlemi."""
    # await asyncio.sleep(0.08)  # Production'da gerçek HTTP RTT olacak
    logger.info(f"[QoD API] SESSION DELETED | sessionId={session_id}")
    return True


class QoDManager:
    """
    Edge AI Hysteresis (Kesiklik) Yöneticisi.
    Anlık dalgalanmaları önler ve gereksiz API isteklerini sönümler.
    Ağ kaynaklarını (5G bandwidth) risk seviyesine göre dinamik olarak açıp kapatır.
    """

    def __init__(self):
        self.active_session_id = None
        self.current_profile = None
        self.high_risk_since = None
        self.low_risk_since = None
        self._lock = asyncio.Lock()  # Asenkron işlem kilit mekanizması
        
        # Süreler
        self.TIME_TO_OPEN_S = 3.0  # Orta risk 3 saniye sürerse THROUGHPUT_S aç
        self.TIME_TO_CLOSE = 2.0   # Risk geçerse 2 saniye sonra kapat

    async def update(self, risk_info: dict, current_time: float):
        """
        Risk hesaplamasına göre QoD durumunu günceller ve gerekirse 
        CAMARA API'yi asenkron çağırır. Birden fazla asenkron çağrının 
        çakışıp aynı anda session açmasını Lock ile engeller.
        """
        async with self._lock:
            score = risk_info.get("score", 0.0)
            force_l = risk_info.get("force_qod_l", False)
            target_profile = risk_info.get("qod_profile", "THROUGHPUT_S")

            # 1. ACİL DURUM (Kabin İhlali) -> Hysteresis atlanır.
            if force_l:
                if self.current_profile != "THROUGHPUT_L":
                    logger.warning(f"[QoDManager] ACIL DURUM TETIKLENDI! Kabin Ihlali tespit edildi. Hysteresis atlandi.")
                    
                    # Eski session varsa kapat
                    if self.active_session_id:
                        await mock_camara_delete_session(self.active_session_id)
                    
                    # Yeni L session aç
                    payload = await mock_camara_create_session("THROUGHPUT_L")
                    self.active_session_id = payload["sessionId"]
                    self.current_profile = "THROUGHPUT_L"
                    
                    # Sayaçları sıfırla
                    self.high_risk_since = None
                    self.low_risk_since = None
                return

            # 2. NORMAL HYSTERESIS (Örn: Plaka okuma ihtiyacı, yakınlık)
            if score >= 60:
                self.low_risk_since = None  # Düşük risk sayacını iptal et
                
                if self.high_risk_since is None:
                    self.high_risk_since = current_time
                
                # Yeterince beklendiyse API'yi çağır
                if current_time - self.high_risk_since >= self.TIME_TO_OPEN_S:
                    if self.current_profile != target_profile:
                        if self.active_session_id:
                            await mock_camara_delete_session(self.active_session_id)
                        payload = await mock_camara_create_session(target_profile)
                        self.active_session_id = payload["sessionId"]
                        self.current_profile = target_profile
            else:
                # Risk düşükse kapatma sayacını başlat
                self.high_risk_since = None
                
                # Sadece aktif bir session varsa kapatma sayacı çalışır
                if self.current_profile is not None:
                    if self.low_risk_since is None:
                        self.low_risk_since = current_time
                    
                    if current_time - self.low_risk_since >= self.TIME_TO_CLOSE:
                        if self.active_session_id:
                            await mock_camara_delete_session(self.active_session_id)
                        self.active_session_id = None
                        self.current_profile = None
                        self.low_risk_since = None
