# silero-tts-for-luna-translator.py

import io, os, sys, time, struct, torch, numpy as np, psutil
from bottle import Bottle, request, response, run
from num2words import num2words
from urllib.parse import unquote
from functools import lru_cache
import threading

# DEBUG = os.environ.get('DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')
DEBUG = True
MAIN_VERSION = "0.2b"

class Config:
    MODEL_PATH = "models/v5_5_ru.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAMPLE_RATE = 48000 # Hz
    HOST, PORT = "127.0.0.1", 5000
    MAX_TEXT_LENGTH = 950 # chars
    BASE_SPEED = 1.0
    CPU_MONITOR_INTERVAL, CPU_SAMPLE_DURATION = 1.0, 0.1 # sec
    CPU_HIGH_THRESHOLD, CPU_CRITICAL_THRESHOLD = 80.0, 95.0 # %
    QUALITY_LEVELS = [
        {"sample_rate": 8000,  "put_accent": False, "put_yo": False, "name": "LOWEST"},
        {"sample_rate": 8000,  "put_accent": True,  "put_yo": False, "name": "LOW"},
        {"sample_rate": 24000, "put_accent": False, "put_yo": False, "name": "MEDIUM"},
        {"sample_rate": 24000, "put_accent": True,  "put_yo": True,  "name": "HIGH"},
        {"sample_rate": 48000, "put_accent": True,  "put_yo": False, "name": "VERY_HIGH"},
        {"sample_rate": 48000, "put_accent": True,  "put_yo": True,  "name": "MAXIMUM"}
    ]

class AudioPauses:
    SENTENCE, COMMA, SENTENCE_END = 320, 220, 400 # ms

# ==================== МАССИВ ГОЛОСОВ И ТОНКИХ НАСТРОЕК ====================
SPEAKERS = [
    {"id": 0, "name": "aidar",   "style": "male",   "lang": ["ru"], "volume_boost": 3,   "pitch": "high",   "base_speed": 1.1},
    {"id": 1, "name": "baya",    "style": "female", "lang": ["ru"], "volume_boost": 0,   "pitch": "low",    "base_speed": 1.0},
    {"id": 2, "name": "kseniya", "style": "female", "lang": ["ru"], "volume_boost": 0,   "pitch": "low",    "base_speed": 1.0},
    {"id": 3, "name": "xenia",   "style": "female", "lang": ["ru"], "volume_boost": 1,   "pitch": "medium", "base_speed": 0.95},
    {"id": 4, "name": "eugene",  "style": "male",   "lang": ["ru"], "volume_boost": 0.5, "pitch": "low",    "base_speed": 0.9},
]

# ==================== CPU МОНИТОР ====================
class CPUMonitor:
    """Мониторинг нагрузки CPU и автоматическое управление качеством генерации"""
    def __init__(self):
        self.current_quality_level = len(Config.QUALITY_LEVELS) - 1
        self.current_load, self.last_change_time, self.max_level = 0.0, 0, len(Config.QUALITY_LEVELS) - 1
        self.lock, self.running, self.load_history, self.max_history_size = threading.Lock(), True, [], 3
        self.min_change_interval = 1.5
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _get_cpu_load(self) -> float:
        try: return psutil.cpu_percent(interval=Config.CPU_SAMPLE_DURATION)
        except: return 0.0
    
    def _add_to_history(self, value: float):
        self.load_history.append(value)
        if len(self.load_history) > self.max_history_size: self.load_history.pop(0)
    
    def _get_average_load(self) -> float:
        return sum(self.load_history) / len(self.load_history) if self.load_history else 0.0
    
    def _calculate_target_quality(self, avg_load: float) -> int:
        if avg_load >= Config.CPU_CRITICAL_THRESHOLD: return 0
        if avg_load >= Config.CPU_HIGH_THRESHOLD:
            load_ratio = (avg_load - Config.CPU_HIGH_THRESHOLD) / (Config.CPU_CRITICAL_THRESHOLD - Config.CPU_HIGH_THRESHOLD)
            return max(0, self.max_level - int(load_ratio * self.max_level))
        return self.max_level
    
    def _monitor_loop(self):
        while self.running:
            try:
                cpu_load = self._get_cpu_load()
                with self.lock:
                    self._add_to_history(cpu_load)
                    avg_load = self._get_average_load()
                    self.current_load = avg_load
                    target_level = self._calculate_target_quality(avg_load)
                    now = time.time()
                    if target_level != self.current_quality_level and now - self.last_change_time >= self.min_change_interval:
                        old_level = self.current_quality_level
                        self.current_quality_level += 1 if target_level > self.current_quality_level else -1
                        self.current_quality_level = max(0, min(self.current_quality_level, self.max_level))
                        self.last_change_time = now
                        if DEBUG:
                            print(f"[CPU] {'↑' if self.current_quality_level > old_level else '↓'} LOAD {avg_load:.1f}% → {Config.QUALITY_LEVELS[old_level]['name']} → {Config.QUALITY_LEVELS[self.current_quality_level]['name']}")
                time.sleep(Config.CPU_MONITOR_INTERVAL)
            except: time.sleep(1)
    
    def get_current_quality_config(self) -> dict:
        with self.lock: return Config.QUALITY_LEVELS[self.current_quality_level].copy()
    
    def get_cpu_load(self) -> float:
        with self.lock: return self.current_load

# ==================== УТИЛИТЫ ====================
@lru_cache(maxsize=1024)
def num_to_words(num: str) -> str:
    """Конвертация числа в текст на русском языке"""
    if not num or not num.isdigit() or len(num) > 9: return str(num) if num else ""
    return num2words(int(num), lang='ru')

class TextProcessor:
    """Обработка текста: конвертация чисел, транслитерация, управление паузами"""
    def __init__(self):
        self.allowed_chars = frozenset("_~абвгдеёжзийклмнопрстуфхцчшщъыьэюя +.,!?…:;–")
        self.latin_letters = frozenset("abcdefghijklmnopqrstuvwxyz")
        self.punctuation_config = {'.': {'pause': AudioPauses.SENTENCE, 'end_sentence': True},
                    '!': {'pause': AudioPauses.SENTENCE, 'end_sentence': True},
                    '?': {'pause': AudioPauses.SENTENCE, 'end_sentence': True},
                    '(': {'pause': AudioPauses.COMMA, 'end_sentence': False},
                    ')': {'pause': AudioPauses.COMMA, 'end_sentence': False},
                    ',': {'pause': AudioPauses.COMMA, 'end_sentence': False},
                    ';': {'pause': AudioPauses.COMMA, 'end_sentence': False},
                    ':': {'pause': AudioPauses.COMMA // 2, 'end_sentence': False}}
        self._init_trie()
    
    def _init_trie(self):
        translit_map = {'ough':'о','augh':'о','eigh':'эй','tion':'шн','shch':'щ',
            'tch':'ч','sch':'ск','scr':'скр','thr':'зр','squ':'скв','ear':'ир',
            'air':'эр','are':'эр','the':'зэ','and':'энд','ea':'и','ee':'и','oo':'у',
            'ai':'эй','ay':'эй','ei':'эй','ey':'эй','oi':'ой','oy':'ой','ou':'ау',
            'ow':'ау','au':'о','aw':'о','ie':'и','ui':'у','ue':'ю','uo':'уо','eu':'ю',
            'ew':'ю','oa':'о','oe':'о','sh':'ш','ch':'ч','zh':'ж','th':'з','kh':'х',
            'ti':'тай','ts':'ц','ph':'ф','wh':'в','gh':'г','qu':'кв','gu':'г','dg':'дж',
            'ce':'це','ci':'си','cy':'си','ck':'к','ge':'дж','gi':'джи','gy':'джи',
            'er':'эр','a':'а','b':'б','c':'к','d':'д','e':'е','f':'ф','g':'г','h':'х',
            'i':'и','j':'дж','k':'к','l':'л','m':'м','n':'н','o':'о','p':'п','q':'к',
            'r':'р','s':'с','t':'т','u':'у','v':'в','w':'в','x':'кс','y':'й','z':'з'}
        self.translit_trie = {}
        for k, v in translit_map.items():
            node = self.translit_trie
            for ch in k:
                node = node.setdefault(ch, {})
            node['_'] = v
    
    def process_text(self, text: str, add_final_pause: bool = True) -> str:
        if not text: return ""
        text, result_parts, i, n = unquote(text).lower(), [], 0, len(text)
        ends_with_sentence, last_was_space, has_latin = False, False, any(ch in self.latin_letters for ch in text)
        while i < n:
            ch = text[i]
            if ch.isdigit():
                i, res = self._process_number(text, i)
                result_parts.append(res)
                ends_with_sentence, last_was_space = False, False
                continue
            if has_latin and ch in self.latin_letters:
                new_i, trans = self._process_transliteration(text, i)
                if trans and trans != ch:
                    result_parts.append(trans)
                    i, ends_with_sentence, last_was_space = new_i, False, False
                    continue
            if ch in self.punctuation_config:
                cfg = self.punctuation_config[ch]
                if result_parts and result_parts[-1] == ' ': result_parts.pop()
                result_parts.extend([ch, f'<break time="{cfg["pause"]}ms"/> '])
                ends_with_sentence, last_was_space, i = cfg['end_sentence'], True, i + 1
                continue
            if ch.isspace() or ch == ' ':
                if not last_was_space: result_parts.append(' ')
                last_was_space, i = True, i + 1
                continue
            if ch in self.allowed_chars:
                result_parts.append(ch)
                ends_with_sentence, last_was_space, i = False, False, i + 1
                continue
            if not last_was_space: result_parts.append(' ')
            last_was_space, i = True, i + 1
        result = ''.join(result_parts).strip()
        if add_final_pause and ends_with_sentence:
            result += f'<break time="{AudioPauses.SENTENCE_END}ms"/>'
        return result
    
    def _process_number(self, text: str, start: int) -> tuple:
        i, n = start, len(text)
        j = i
        while j < n and text[j].isdigit(): j += 1
        if j == i: return i + 1, text[i]
        num1 = text[i:j]
        if j < n and text[j] in '.,' and j + 1 < n and text[j + 1].isdigit():
            k = j + 1
            while k < n and text[k].isdigit(): k += 1
            return k, f"{num_to_words(num1)} точка {num_to_words(text[j+1:k])}"
        if j < n and text[j] == '/' and j + 1 < n and text[j + 1].isdigit():
            k = j + 1
            while k < n and text[k].isdigit(): k += 1
            return k, f"{num_to_words(num1)} дробь {num_to_words(text[j+1:k])}"
        return j, num_to_words(num1)
    
    def _process_transliteration(self, text: str, pos: int) -> tuple:
        node, best_match, best_pos = self.translit_trie, None, pos
        j = pos
        while j < len(text) and text[j] in node:
            node = node[text[j]]
            j += 1
            if '_' in node: best_match, best_pos = node['_'], j
        return (best_pos, best_match) if best_match else (pos + 1, text[pos] if text[pos] in self.allowed_chars else " ")

# ==================== АУДИО ПРОЦЕССОР ====================

class AudioProcessor:
    """Генерация аудио через Silero TTS с адаптивным качеством"""
    def __init__(self, model, device, cpu_monitor):
        self.model, self.device, self.cpu_monitor = model, device, cpu_monitor
        self.text_processor = TextProcessor()
    
    def _numpy_to_wav_bytes(self, audio_np: np.ndarray, sample_rate: int) -> bytes:
        """Конвертирует numpy массив в WAV bytes"""
        audio_int16 = (audio_np * 32767).astype(np.int16)
        buffer = io.BytesIO()
        buffer.write(b'RIFF')
        buffer.write(struct.pack('<I', 36 + len(audio_int16) * 2))
        buffer.write(b'WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00')
        buffer.write(struct.pack('<I', sample_rate))
        buffer.write(struct.pack('<I', sample_rate * 2))
        buffer.write(struct.pack('<H', 2))
        buffer.write(struct.pack('<H', 16))
        buffer.write(b'data')
        buffer.write(struct.pack('<I', len(audio_int16) * 2))
        buffer.write(audio_int16.tobytes())
        return buffer.getvalue()
    
    def synthesize(self, text: str, speaker_id: int, length: float, pitch_adjustment: float) -> bytes:
        start_time = time.time() if DEBUG else None
        try:
            quality = self.cpu_monitor.get_current_quality_config()
            sample_rate = min(Config.SAMPLE_RATE, quality["sample_rate"])
            speaker = SPEAKERS[speaker_id]["name"] if 0 <= speaker_id < len(SPEAKERS) else SPEAKERS[0]["name"]
            voice = {k: v for k, v in SPEAKERS[speaker_id].items() if k in ['volume_boost', 'pitch', 'base_speed']}
            
            param_speed = (1 - length) * (5 if length >= 1 else 15)
            final_speed = max(0.1, min(10.0, voice["base_speed"] * (2 ** (param_speed / 10))))
            pitch_map = {"x-low": -10, "low": -4, "medium": 0, "high": 4, "x-high": 10}
            base_pitch = pitch_map.get(voice["pitch"], 0)
            final_pitch = max(-10, min(10, base_pitch + pitch_adjustment))
            pitch_str = [k for k, v in pitch_map.items() if v == final_pitch][0] if final_pitch in pitch_map.values() else "medium"
            
            processed = self.text_processor.process_text(text) or text
            ssml = f'<speak><prosody rate="{int(final_speed*100)}%" pitch="{pitch_str}">{processed}</prosody></speak>'
            
            if DEBUG:
                print(f"[DEBUG] Quality: {quality['name']} | CPU: {self.cpu_monitor.get_cpu_load():.1f}% | SR: {sample_rate}Hz")
                print(f"[DEBUG] Speaker: {speaker} | Speed: {final_speed:.2f}x | Pitch: {pitch_str}")
            
            audio = self.model.apply_tts(ssml_text=ssml, speaker=speaker, sample_rate=sample_rate, put_accent=quality["put_accent"], put_yo=quality["put_yo"])
            audio_np = audio.cpu().numpy() if hasattr(audio, 'cpu') else np.array(audio, dtype=np.float32)
            
            if voice["volume_boost"] != 0:
                volume_factor = 10 ** (voice["volume_boost"] / 20.0)
                audio_np = np.clip(audio_np * volume_factor, -1.0, 1.0)
            
            result = self._numpy_to_wav_bytes(audio_np, sample_rate)
            
            if DEBUG and start_time:
                print(f"[DEBUG] Total: {(time.time()-start_time)*1000:.2f}ms | Audio: {len(audio_np)/sample_rate:.2f}s | RTF: {((time.time()-start_time)*1000) / (len(audio_np)/sample_rate*1000):.2f}x")
                print(f"[DEBUG] Text length: {len(text)}, text: {text}")
            return result
        except Exception as e:
            print(f"[X] Synthesis error: {e}")
            silent = np.zeros(int(16000 * 0.1), dtype=np.float32)
            return self._numpy_to_wav_bytes(silent, 16000)

# ==================== HTTP СЕРВЕР ====================
class TTServer:
    """HTTP сервер для Silero TTS с REST API"""
    def __init__(self):
        self.app, self.model, self.cpu_monitor, self.audio_processor = Bottle(), None, None, None
        self.setup_routes()
    
    def load_model(self):
        if not os.path.exists(Config.MODEL_PATH):
            print(f"\n[X] Model not found: {Config.MODEL_PATH}")
            sys.exit(1)
        print(f"[*] Loading model '{Config.MODEL_PATH}'... ({Config.DEVICE})")
        package = torch.package.PackageImporter(Config.MODEL_PATH)
        self.model = package.load_pickle("tts_models", "model")
        self.model.to(Config.DEVICE)
        self.cpu_monitor = CPUMonitor()
        self.audio_processor = AudioProcessor(self.model, Config.DEVICE, self.cpu_monitor)
        print("[V] Model loaded")
    
    def setup_routes(self):
        @self.app.route('/voice/speakers', method='GET')
        def get_speakers():
            response.content_type = 'application/json'
            return {"vits": [{"id": s["id"], "name": s["name"], "lang": s["lang"]} for s in SPEAKERS]}
        
        @self.app.route('/voice/vits', method='GET')
        def synthesize():
            """Генерирует WAV аудио из текста с заданными параметрами голоса"""
            text, speaker_id = request.query.text or "", int(request.query.id or 0)
            length, pitch = float(request.query.length or 1.0), float(request.query.pitch or 0)
            if not text: return {"error": "Text is required"}, 400
            if len(text) > Config.MAX_TEXT_LENGTH:
                text = text[:Config.MAX_TEXT_LENGTH]
                print(f"[WARNING] Text truncated to {Config.MAX_TEXT_LENGTH} chars")
            try:
                audio_data = self.audio_processor.synthesize(text, speaker_id, length, pitch)
                response.content_type = 'audio/wav'
                return audio_data
            except Exception as e:
                print(f"[X] Synthesis failed: {e}")
                return {"error": "Synthesis failed"}, 500
    
    def run(self):
        self.load_model()
        print('=' * 70)
        print(f" Silero TTS Server for LunaTranslator v{MAIN_VERSION}")
        print(f" Device: {str(Config.DEVICE).upper()} | http://{Config.HOST}:{Config.PORT}")
        print(f" Endpoints: GET /voice/speakers | GET /voice/vits?text=&id=&length=&pitch=")
        print(f" CPU thresholds: {Config.CPU_HIGH_THRESHOLD}% (high), {Config.CPU_CRITICAL_THRESHOLD}% (critical)")
        print('=' * 70)
        run(self.app, host=Config.HOST, port=Config.PORT, quiet=False)

if __name__ == "__main__":
    TTServer().run()