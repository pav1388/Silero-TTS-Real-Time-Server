# selfbuild_tts.py
# pav13

from myutils.config import urlpathjoin
from tts.basettsclass import TTSbase, SpeechParam
import time


class TTS(TTSbase):
    arg_support_pitch = True
    arg_support_speed = True
    
    SILERO_SERVER_URL = "http://127.0.0.1:23457"
    SILERO_STREAMING = True
    
    SILERO_REAL_SPEAKERS_COUNT = 0
    SILERO_MALE_SPEAKER_IDS = []
    SILERO_FEMALE_SPEAKER_IDS = []
    SILERO_VOICE_PRESETS = {
            0: {"vol_boost": 0, "base_speed": 119, "base_pitch": "high"},    # aidar
            1: {"vol_boost": 1.5, "base_speed": 100, "base_pitch": "low"},   # baya
            2: {"vol_boost": 0, "base_speed": 100, "base_pitch": "medium"},  # kseniya
            3: {"vol_boost": -1, "base_speed": 89, "base_pitch": "high"},    # eugene
            4: {"vol_boost": 0.5, "base_speed": 100, "base_pitch": "medium"} # xenia
    }
    
    def getvoicelist(self):
        headers = {"ngrok-skip-browser-warning": "true"}
        self.SILERO_REAL_SPEAKERS_COUNT = 0
        try:
            response = self.proxysession.get(
                urlpathjoin(self.SILERO_SERVER_URL, "/speakers"), 
                headers=headers,
                timeout=3
            )
            
            if response.status_code != 200:
                return [], []
            
            data = response.json()
            speakers = data.get("silero", [])
            
            if not speakers:
                return [], []
            
        except Exception as e:
            print("[Silero TTS] Can't connect to server: {}".format(e))
            return [], []
        
        voicelist = []
        internal = []
        
        for s in speakers:
            speaker_id = s.get("id", 0)
            speaker_name = s.get("name", "unknown")
            speaker_lang = s.get("lang", "n/d")
            speaker_gender = s.get("gender", "n/d")
            model_info = "silero_{}_{}_{}_{}".format(
                speaker_id, speaker_name, speaker_lang, speaker_gender
            )
            
            if speaker_name not in ['RANDOM', 'RANDOM_M', 'RANDOM_F', 'HASH']:
                self.SILERO_REAL_SPEAKERS_COUNT += 1
                if speaker_gender == "male":
                    self.SILERO_MALE_SPEAKER_IDS.append(speaker_id)
                elif speaker_gender == "female":
                    self.SILERO_FEMALE_SPEAKER_IDS.append(speaker_id)
            
            voicelist.append(model_info)
            internal.append(("silero", speaker_id, speaker_name))
        
        return internal, voicelist

    def speak(self, content, voice, param: SpeechParam):
        if not content or not content.strip():
            return None
        
        _, speaker_id, _ = voice
        
        # Обработка специальных speaker_id (RANDOM, HASH)
        if speaker_id == self.SILERO_REAL_SPEAKERS_COUNT:  # RANDOM both
            speaker_id = int(time.time() * 100000) % self.SILERO_REAL_SPEAKERS_COUNT
        elif speaker_id == self.SILERO_REAL_SPEAKERS_COUNT + 1:  # RANDOM only_male
            if self.SILERO_MALE_SPEAKER_IDS:
                speaker_id = self.SILERO_MALE_SPEAKER_IDS[
                    int(time.time() * 100000) % len(self.SILERO_MALE_SPEAKER_IDS)]
            else:
                speaker_id = int(time.time() * 100000) % self.SILERO_REAL_SPEAKERS_COUNT
        elif speaker_id == self.SILERO_REAL_SPEAKERS_COUNT + 2:  # RANDOM only_female
            if self.SILERO_FEMALE_SPEAKER_IDS:
                speaker_id = self.SILERO_FEMALE_SPEAKER_IDS[
                    int(time.time() * 100000) % len(self.SILERO_FEMALE_SPEAKER_IDS)]
            else:
                speaker_id = int(time.time() * 100000) % self.SILERO_REAL_SPEAKERS_COUNT
        elif speaker_id == self.SILERO_REAL_SPEAKERS_COUNT + 3:  # HASH both
            t = content[:500]
            h = 5381
            for c in t:
                h = ((h << 5) + h) ^ ord(c)
            speaker_id = (h & 0x7fffffff) % self.SILERO_REAL_SPEAKERS_COUNT
        
        preset = self.SILERO_VOICE_PRESETS.get(speaker_id, {"vol_boost": 0, "base_speed": 100, "base_pitch": "medium"})
        
        # Конвертация speed: из [-10, 10] в проценты [40%, 300%]
        speed_percent = int(preset["base_speed"] + param.speed * (20 if param.speed > 0 else 6))
        speed_percent = max(40, min(300, speed_percent))
        
        # Конвертация pitch: из [-10, 10] в уровни
        pitch_levels = ["x-low", "low", "medium", "high", "x-high"]
        base_idx = pitch_levels.index(preset["base_pitch"]) if preset["base_pitch"] in pitch_levels else 2
        delta = int(round((param.pitch + 10) / 20 * 4)) - 2
        pitch_index = max(0, min(4, base_idx + delta))
        pitch_level = pitch_levels[pitch_index]
        
        response = self.proxysession.get(
            urlpathjoin(self.SILERO_SERVER_URL, "/speak/stream" if self.SILERO_STREAMING else "/speak"),
            params={
                "text": content,
                "id": speaker_id,
                "speed": speed_percent,
                "pitch": pitch_level,
                "vol_boost": preset["vol_boost"]
            },
            headers={"ngrok-skip-browser-warning": "true"},
            stream=True,
            timeout=120 if self.SILERO_STREAMING else 30
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Server error {response.status_code}: {response.text}")
        
        return response