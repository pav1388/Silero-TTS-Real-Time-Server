# selfbuild_tts.py
# pav13

from myutils.config import urlpathjoin
from tts.basettsclass import TTSbase, SpeechParam
import time


class TTS(TTSbase):
    arg_support_pitch = True
    arg_support_speed = True
    
    SILERO_URL = "http://127.0.0.1:23457"
    STREAMING_MODE = True
    
    def init(self):
        self.voice_presets = {
            0: {"vol_boost": 0,   "base_speed": 120, "base_pitch": "high"},    # aidar
            1: {"vol_boost": 1.5, "base_speed": 100, "base_pitch": "low"},     # baya
            2: {"vol_boost": 0,   "base_speed": 100, "base_pitch": "medium"},  # kseniya
            3: {"vol_boost": 0.5, "base_speed": 100, "base_pitch": "medium"},  # xenia
            4: {"vol_boost": -1,  "base_speed": 87,  "base_pitch": "high"}     # eugene
        }

    def getvoicelist(self):
        headers = {"ngrok-skip-browser-warning": "true"}
        
        try:
            response = self.proxysession.get(
                urlpathjoin(self.SILERO_URL, "/silero/speakers"), 
                headers=headers,
                timeout=3
            )
            
            if response.status_code == 200:
                data = response.json()
                speakers = data.get("silero", [])
                voicelist = []
                internal = []
                
                for s in speakers:
                    model_info = "silero_{}_{}_{}_{}".format(s["id"], s["name"], s["lang"], s["gender"])
                    voicelist.append(model_info)
                    internal.append(("silero", s["id"], s["name"]))
                
                return internal, voicelist
        except Exception as e:
            print("[Silero TTS] Can't connect to server: {}".format(e))

    def speak(self, content, voice, param: SpeechParam):
        if not content or not content.strip():
            return None
        
        _, speaker_id, _ = voice
        
        # Чтобы применялся правильный пресет к случайному голосу
        if speaker_id == 5:  # RANDOM both
            speaker_id = int(time.time() * 1000000) % 5
        elif speaker_id == 6:  # RANDOM only_male
            speaker_id = [0, 4][int(time.time() * 1000000) % 2]
        elif speaker_id == 7:  # RANDOM only_female
            speaker_id = [1, 2, 3][int(time.time() * 1000000) % 3]
        elif speaker_id == 8:  # HASH both
            t = text[:200]
            h = 5381
            for c in t:
                h = ((h << 5) + h) ^ ord(c)
            speaker_id = (h & 0x7fffffff) % 5
        
        preset = self.voice_presets.get(speaker_id, {"vol_boost": 0, "base_speed": 100, "base_pitch": "medium"})
        
        # Конвертация speed: из [-10, 10] в проценты [40%, 300%], где 0 = 100% с учетом базовой скорости голоса
        speed_percent = int(preset["base_speed"] + param.speed * (20 if param.speed > 0 else 6))
        speed_percent = max(40, min(300, speed_percent))
        
        # Конвертация pitch: из [-10, 10] в уровни с учетом базовой высоты голоса
        pitch_levels = ["x-low", "low", "medium", "high", "x-high"]
        base_idx = pitch_levels.index(preset["base_pitch"]) if preset["base_pitch"] in pitch_levels else 2
        delta = int(round((param.pitch + 10) / 20 * 4)) - 2  # -2..+2
        pitch_index = max(0, min(4, base_idx + delta))
        pitch_level = pitch_levels[pitch_index]
        
        response = self.proxysession.get(
            urlpathjoin(self.SILERO_URL, "/silero/speak"),
            params={
                "text": content,
                "id": speaker_id,
                "speed": speed_percent,
                "pitch": pitch_level,
                "vol_boost": preset["vol_boost"],
                "stream": str(self.STREAMING_MODE).lower()
            },
            headers={"ngrok-skip-browser-warning": "true"},
            stream=True,
            timeout=120 if self.STREAMING_MODE else 30
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Server error {response.status_code}: {response.text}")
        
        return response