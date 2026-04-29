# silero-tts-for-luna-translator.py
# pav13

import io, os, sys, time, struct, torch, psutil, numpy as np
from bottle import Bottle, request, response, run
from num2words import num2words
from urllib.parse import unquote
from functools import lru_cache
import threading

# DEBUG = os.environ.get('DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')
DEBUG = True
MAIN_VERSION = "0.2.2-dev"

# Конфигурация
class Config:
    MODEL_PATH = "models/v5_5_ru.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAMPLE_RATE = 48000 # 8000 24000 48000
    HOST, PORT = "127.0.0.1", 23456
    MAX_TEXT_LENGTH = 950
    CPU_IDLE_TIMEOUT, CPU_MONITOR_INTERVAL, CPU_SAMPLE_DURATION = 30.0, 1.0, 0.07
    CPU_HIGH_THRESHOLD, CPU_CRITICAL_THRESHOLD = 85.0, 95.0
    QUALITY_LEVELS = [
        {"sample_rate": 8000,  "put_accent": False, "put_yo": False, "name": "LOWEST"},
        {"sample_rate": 8000,  "put_accent": True,  "put_yo": False, "name": "LOW"},
        {"sample_rate": 24000, "put_accent": False, "put_yo": False, "name": "MEDIUM-LOW"},
        {"sample_rate": 24000, "put_accent": True,  "put_yo": True,  "name": "MEDIUM"},
        {"sample_rate": 48000, "put_accent": True,  "put_yo": False, "name": "HIGH"},
        {"sample_rate": 48000, "put_accent": True,  "put_yo": True,  "name": "MAXIMUM"}
    ]

# Список доступных голосов и индивидуальных настроек
SPEAKERS = [
    {"id": 0, "name": "aidar",   "style": "male",   "lang": ["ru"], "volume_boost": 3,   "pitch": "high",   "base_speed": 1.1},
    {"id": 1, "name": "baya",    "style": "female", "lang": ["ru"], "volume_boost": 0,   "pitch": "low",    "base_speed": 1.0},
    {"id": 2, "name": "kseniya", "style": "female", "lang": ["ru"], "volume_boost": 0,   "pitch": "low",    "base_speed": 1.0},
    {"id": 3, "name": "xenia",   "style": "female", "lang": ["ru"], "volume_boost": 1,   "pitch": "medium", "base_speed": 0.95},
    {"id": 4, "name": "eugene",  "style": "male",   "lang": ["ru"], "volume_boost": 0.5, "pitch": "low",    "base_speed": 0.9},
]

# Мониторинг загрузки CPU для динамического изменения качества генерации
class CPUMonitor:
    def __init__(self):
        self.current_quality_level = len(Config.QUALITY_LEVELS) - 1
        self.current_load, self.last_change_time, self.max_level = 0.0, 0, len(Config.QUALITY_LEVELS) - 1
        self.lock, self.running, self.load_history, self.max_history_size = threading.Lock(), False, [], 3
        self.min_change_interval = 1.5
        self.monitor_thread = None
        self.last_activity_time = 0
    
    # Запуск мониторинга CPU
    def start_monitoring(self):
        with self.lock:
            if self.running: return
            self.last_activity_time = time.time()
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            if DEBUG: print("[DEBUG] CPU monitoring ON")
    
    # Остановка мониторинга CPU
    def stop_monitoring(self):
        with self.lock:
            if not self.running: return
            self.running = False
            if DEBUG: print("[DEBUG] CPU monitoring OFF")
            
    # Запись активности сервера
    def record_activity(self):
        self.last_activity_time = time.time()
        if not self.running: self.start_monitoring()
    
    # Проверка простоя и остановка мониторинга
    def _check_idle_and_stop(self):
        if time.time() - self.last_activity_time >= Config.CPU_IDLE_TIMEOUT:
            self.stop_monitoring()
            return True
        return False
    
    # Получение текущей загрузки CPU
    def _get_cpu_load(self) -> float:
        try: return psutil.cpu_percent(interval=Config.CPU_SAMPLE_DURATION)
        except: return 0.0
    
    # Добавление значения в историю загрузки
    def _add_to_history(self, value: float):
        self.load_history.append(value)
        if len(self.load_history) > self.max_history_size: self.load_history.pop(0)
    
    # Получение средней загрузки из истории
    def _get_average_load(self) -> float:
        return sum(self.load_history) / len(self.load_history) if self.load_history else 0.0
    
    # Расчет целевого уровня качества на основе загрузки
    def _calculate_target_quality(self, avg_load: float) -> int:
        if avg_load >= Config.CPU_CRITICAL_THRESHOLD: return 0
        if avg_load >= Config.CPU_HIGH_THRESHOLD:
            load_ratio = (avg_load - Config.CPU_HIGH_THRESHOLD) / (Config.CPU_CRITICAL_THRESHOLD - Config.CPU_HIGH_THRESHOLD)
            return max(0, self.max_level - int(load_ratio * self.max_level))
        return self.max_level
    
    # Основной цикл мониторинга
    def _monitor_loop(self):
        while self.running:
            try:
                if self._check_idle_and_stop(): break
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
                        if DEBUG: print(f"[DEBUG] Quality {'UP' if self.current_quality_level > old_level else 'DOWN'} to Lvl {self.current_quality_level} LOAD {avg_load:.1f}%")
                time.sleep(Config.CPU_MONITOR_INTERVAL)
            except: time.sleep(5)
    
    # Получение текущей конфигурации качества
    def get_current_quality_config(self) -> dict:
        with self.lock: return Config.QUALITY_LEVELS[self.current_quality_level].copy()
    
    # Получение текущей загрузки CPU
    def get_cpu_load(self) -> float:
        with self.lock: return self.current_load

# Преобразования чисел в слова (+ кэширование)
@lru_cache(maxsize=512)
def num_to_words(num: str) -> str:
    if not num or not num.isdigit() or len(num) > 9: return str(num) if num else ""
    return num2words(int(num), lang='ru')

# Обработка текста с преобразованием в SSML
class TextProcessor:
    BREAK_TIME_MAP = {'.': 360, ',': 230, '!': 420, '?': 420, '(': 230, ')': 230, '[': 230, ']': 230, ':': 230, ';': 230, '-': 230}
    ALLOWED = frozenset("_~абвгдеёжзийклмнопрстуфхцчшщъыьэюя +.,!?…:;–")
    LATIN = frozenset("abcdefghijklmnopqrstuvwxyz")
    TRANSLIT_MAP = {'ough':'о','augh':'о','eigh':'эй','igh':'ай','tion':'шн','shch':'щ','tch':'ч','sch':'ск','scr':'скр','thr':'зр','squ':'скв','ear':'ир','air':'эр','are':'эр','the':'зэ','and':'энд','ea':'и','ee':'и','oo':'у','ai':'эй','ay':'эй','ei':'эй','ey':'эй','oi':'ой','oy':'ой','ou':'ау','ow':'ау','au':'о','aw':'о','ie':'и','ui':'у','ue':'ю','uo':'уо','eu':'ю','ew':'ю','oa':'о','oe':'о','sh':'ш','ch':'ч','zh':'ж','th':'з','kh':'х','ti':'тай','ts':'ц','ph':'ф','wh':'в','gh':'г','qu':'кв','gu':'г','dg':'дж','ce':'це','ci':'си','cy':'си','ck':'к','ge':'дж','gi':'джи','gy':'джи','er':'эр','a':'а','b':'б','c':'к','d':'д','e':'е','f':'ф','g':'г','h':'х','i':'и','j':'дж','k':'к','l':'л','m':'м','n':'н','o':'о','p':'п','q':'к','r':'р','s':'с','t':'т','u':'у','v':'в','w':'в','x':'кс','y':'й','z':'з'}

    def __init__(self):
        self.base_speed = 1.0
        self.base_pitch = "medium"
        self.trie = {}
        for k, v in self.TRANSLIT_MAP.items():
            node = self.trie
            for ch in k:
                node = node.setdefault(ch, {})
            node['_'] = v

    # Установка параметров SSML (скорость и высота тона)
    def set_ssml_params(self, speed: float, pitch: str):
        self.base_speed, self.base_pitch = speed, pitch

    # Корректировка высоты тона
    def _adj_pitch(self, pitch: str, delta: int) -> str:
        order = ["x-low", "low", "medium", "high", "x-high"]
        try:
            idx = max(0, min(len(order) - 1, order.index(pitch) + delta))
            return order[idx]
        except ValueError: return "medium"

    # Основной метод
    def process_text(self, text: str) -> str:
        if not text: return ""
        len_text = len(text)
        if len_text > Config.MAX_TEXT_LENGTH:
            len_text = Config.MAX_TEXT_LENGTH
            text = text[:len_text]
            if DEBUG: print(f"[DEBUG] Text length truncated to {len_text} chars.")
        text = unquote(text).lower()
        has_latin = any(ch in self.LATIN for ch in text)
        return f'<speak>{self._proc(text, len_text, has_latin)}</speak>'

    # Обработка текста
    def _proc(self, text: str, len_text: int, has_latin: bool) -> str:
        res, buf, i, n = [], [], 0, len_text
        while i < n:
            ch = text[i]
            if ch.isdigit():
                i, p = self._num(text, i); buf.append(p); continue
            if has_latin and ch in self.LATIN:
                ni, tr = self._trans(text, i)
                if tr and tr != ch: buf.append(tr); i = ni; continue
            if ch in self.BREAK_TIME_MAP:
                s = ''.join(buf).strip()
                if s:
                    keep = ch in '!?'
                    res.append(self._wrap(s + (ch if keep else ""), ch))
                res.append(f'<break time="{self.BREAK_TIME_MAP[ch]}ms"/>')
                buf.clear(); i += 1; continue
            if ch.isspace() or ch == ' ':
                if not buf or buf[-1] != ' ': buf.append(' ')
                i += 1; continue
            if ch in self.ALLOWED:
                buf.append(ch); i += 1; continue
            if not (buf and buf[-1] == ' '): buf.append(' ')
            i += 1
        if buf:
            s = ''.join(buf).strip()
            if s: res.append(self._wrap(s, None))
        return ''.join(res).strip()

    # Обработка чисел в тексте
    def _num(self, text: str, start: int) -> tuple:
        i, n = start, len(text)
        while i < n and text[i].isdigit(): i += 1
        if i == start: return start + 1, text[start]
        num1 = text[start:i]
        if i < n and text[i] in '.,' and i + 1 < n and text[i+1].isdigit():
            k = i + 1
            while k < n and text[k].isdigit(): k += 1
            return k, f"{num_to_words(num1)} точка {num_to_words(text[i+1:k])}"
        if i < n and text[i] == '/' and i + 1 < n and text[i+1].isdigit():
            k = i + 1
            while k < n and text[k].isdigit(): k += 1
            return k, f"{num_to_words(num1)} дробь {num_to_words(text[i+1:k])}"
        return i, num_to_words(num1)

    # Транслитерация латинских символов
    def _trans(self, text: str, pos: int) -> tuple:
        node, best, best_pos = self.trie, None, pos
        j = pos
        while j < len(text) and text[j] in node:
            node = node[text[j]]; j += 1
            if '_' in node: best, best_pos = node['_'], j
        return (best_pos, best) if best else (pos + 1, text[pos] if text[pos] in self.ALLOWED else " ")

    # Оборачивание текста в SSML теги с параметрами для придания
    # эмоциональных оттенков восклицания и вопрошения
    def _wrap(self, txt: str, end_punct: str) -> str:
        if not txt: return ""
        def attrs(rate, pitch): return f'rate="{rate}%" pitch="{pitch}"'
        base_r = f"{int(self.base_speed * 100)}"
        base_p = self.base_pitch
        cfg = {'!': (115, 1), '?': (95, 1)}
        if end_punct in cfg:
            sm, pd = cfg[end_punct]
            sr, sp = f"{int(self.base_speed * sm)}", self._adj_pitch(base_p, pd)
            txt = txt[:-1] + ' ' + end_punct
            words = txt.split()
            if len(words) < 3: return f'<prosody {attrs(sr, sp)}>{txt}</prosody>'
            return (f'<prosody {attrs(base_r, base_p)}>{" ".join(words[:-2])} </prosody>'
                    f'<prosody {attrs(sr, sp)}>{" ".join(words[-2:])}</prosody>')
        return f'<prosody {attrs(base_r, base_p)}>{txt}</prosody>'

# Обработка аудио (синтез речи)
class AudioProcessor:
    def __init__(self, model, device, cpu_monitor):
        self.model, self.device, self.cpu_monitor = model, device, cpu_monitor
        self.text_processor = TextProcessor()

    # Преобразование тензора в WAV формат
    def _to_wav(self, t, sr):
        d = t.detach().cpu().numpy().squeeze()
        raw = np.clip(d * 32767, -32768, 32767).astype(np.int16).tobytes()
        del d
        sz = len(raw)
        hdr = b'RIFF' + struct.pack('<I', 36 + sz) + b'WAVEfmt '
        hdr += struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16)
        hdr += b'data' + struct.pack('<I', sz)
        return hdr + raw

    # Основной метод
    def synthesize(self, text, speaker_id, length, pitch_adj):
        t_start = time.time() if DEBUG else None
        try:
            self.cpu_monitor.record_activity()
            q = self.cpu_monitor.get_current_quality_config()
            sr = min(Config.SAMPLE_RATE, q["sample_rate"])
            
            if not (0 <= speaker_id < len(SPEAKERS)): 
                raise ValueError(f"Bad Speaker ID: {speaker_id}")
            
            spk = SPEAKERS[speaker_id]
            speed = max(0.1, min(10.0, spk['base_speed'] * (2 ** ((1 - length) * (5 if length >= 1 else 15) / 10))))
            p_map = {"x-low": -10, "low": -4, "medium": 0, "high": 4, "x-high": 10}
            base_p = p_map.get(spk['pitch'], 0)
            final_p = max(-10, min(10, base_p + pitch_adj))
            p_str = next((k for k, v in p_map.items() if v == final_p), "medium")
            
            self.text_processor.set_ssml_params(speed, p_str)
            ssml = self.text_processor.process_text(text)
            
            if DEBUG: 
                print(f"[DEBUG] Quality: {q['name']} Spk:{spk['name']} Speed:{speed:.2f} Pitch:{p_str} SR:{sr}")
                print(f"[DEBUG] Text len:{len(text)} SSML:{ssml}")
            with torch.no_grad():
                audio = self.model.apply_tts(ssml_text=ssml, speaker=spk['name'], 
                    sample_rate=sr, put_accent=q['put_accent'], put_yo=q['put_yo'])
            
            if audio.dim() == 1: audio = audio.unsqueeze(0)
            if spk['volume_boost'] != 0:
                audio = torch.clamp(audio * (10 ** (spk['volume_boost'] / 20.0)), -1.0, 1.0)

            wav_bytes = self._to_wav(audio, sr)
            del audio
                
            if DEBUG and t_start:
                num_samples = len(wav_bytes) - 44
                dur = num_samples / (sr * 2)
                if dur > 0: print(f"[DEBUG] Time:{(time.time()-t_start)*1000:.0f}ms Dur:{dur:.2f}s RTF:{(time.time()-t_start)/dur:.2f}")
            
            return wav_bytes
        except Exception as e:
            print(f"[X] Error: {e}")
            if audio is not None: del audio
            raise e

# TTS сервер (однопоточный)
class TTSServer:
    def __init__(self):
        self.app, self.model, self.cpu_monitor, self.audio_processor = Bottle(), None, None, None
        self.setup_routes()
    
    # Загрузка модели TTS
    def load_model(self):
        url = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
        if not os.path.exists(Config.MODEL_PATH):
            print(f"\n[!] Model not found: {Config.MODEL_PATH}")
            os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
            try:
                import urllib.request
                print(f"[*] Downloading v5_5_ru model...")
                urllib.request.urlretrieve(url, Config.MODEL_PATH, 
                    lambda b, bs, ts: print(f"\r {int(b*bs*100/ts)}%", end=''))
                print(f"\n[V] Model downloaded.")
            except Exception as e:
                print(f"\n[X] Download failed: {e}")
                print(f"\n Please download manually from: {url}")
                input(f" and save as: {Config.MODEL_PATH}")
                sys.exit(1)
        if Config.DEVICE.type == 'cpu':
            try: 
                cores = psutil.cpu_count(logical=False)
                if cores is None: cores = os.cpu_count()
            except: 
                cores = os.cpu_count()
            n_threads = 2 if (cores and cores > 1) else 1
            torch.set_num_threads(n_threads)
            torch.set_num_interop_threads(1)
            print(f"[INFO] CPU Threads set to: {n_threads} (Detected cores: {cores})")
        print(f"[*] Loading model '{Config.MODEL_PATH}'... ({Config.DEVICE})")
        try:
            package = torch.package.PackageImporter(Config.MODEL_PATH)
            self.model = package.load_pickle("tts_models", "model")
            self.model.to(Config.DEVICE)
        except Exception as e:
            print(f"[X] Failed to load model. File might be corrupted. Delete {Config.MODEL_PATH} and restart.")
            raise e

        self.cpu_monitor = CPUMonitor()
        self.audio_processor = AudioProcessor(self.model, Config.DEVICE, self.cpu_monitor)
        print("[V] Model loaded successfully")
    
    # Настройка маршрутов API
    def setup_routes(self):
        @self.app.route('/voice/speakers', method='GET')
        def get_speakers():
            response.content_type = 'application/json'
            print("[HTTP] Request list of speakers from LunaTranslator.")
            return {"vits": [{"id": s["id"], "name": s["name"], "lang": s["lang"]} for s in SPEAKERS]}
        
        @self.app.route('/voice/vits', method='GET')
        def synthesize():
            text, speaker_id = request.query.text or "", int(request.query.id or 0)
            length, pitch = float(request.query.length or 1.0), float(request.query.pitch or 0)
            if not text: return {"error": "Text is required"}, 400
            print(f"[HTTP] New text request.")
            try:
                audio_data = self.audio_processor.synthesize(text, speaker_id, length, pitch)
                response.content_type = 'audio/wav'
                return audio_data
            except Exception as e:
                print(f"[X] Synthesis failed: {e}")
                return {"error": str(e)}, 500
    
    # Запуск сервера
    def run(self):
        self.load_model()
        print('=' * 60)
        print(f" Silero TTS Server for LunaTranslator v{MAIN_VERSION}")
        print(f" Device: {str(Config.DEVICE).upper()} | http://{Config.HOST}:{Config.PORT}")
        if DEBUG: print(" DEBUG mode ON")
        print('=' * 60)
        print("Press Ctrl-C to quit.")
        run(self.app, host=Config.HOST, port=Config.PORT, quiet=True)

# Точка входа
if __name__ == "__main__":
    TTSServer().run()