# silero-tts-for-luna-translator.py
# pav13

import os, sys, time, ctypes, struct, psutil, signal, torch, numpy as np
from bottle import Bottle, request, response, run, hook
from num2words import num2words
from urllib.parse import unquote
from functools import lru_cache
import threading

MAIN_VERSION = "0.4.4-dev"
# DEBUG = os.environ.get('DEBUG', '0').lower() in ('1', 'true', 'yes', 'on')
DEBUG = True

# Список доступных голосов и индивидуальных подстроек
SPEAKERS = [
    {"id": 0, "name": "aidar",   "style": "male",   "lang": ["ru"], "vol_boost": 0,   "pitch": "high",   "base_speed": 1.2},
    {"id": 1, "name": "baya",    "style": "female", "lang": ["ru"], "vol_boost": 1.5,   "pitch": "low",    "base_speed": 1.0},
    {"id": 2, "name": "kseniya", "style": "female", "lang": ["ru"], "vol_boost": 0,   "pitch": "medium",    "base_speed": 1.0},
    {"id": 3, "name": "xenia",   "style": "female", "lang": ["ru"], "vol_boost": 0.5,   "pitch": "medium", "base_speed": 1.0},
    {"id": 4, "name": "eugene",  "style": "male",   "lang": ["ru"], "vol_boost": -1, "pitch": "high",    "base_speed": 0.87},
]


class Config:
    """Конфигурация приложения"""
    MODEL_PATH = "models/v5_5_ru.pt"
    URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
    TORCH_DEVICE = os.environ.get('TORCH_DEVICE', 'cpu').lower()
    if TORCH_DEVICE in ('cuda', 'gpu'):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device('cpu')
    SAMPLE_RATE = 48000
    HOST, PORT = "127.0.0.1", 23456
    MAX_TEXT_LENGTH = 800
    CPU_IDLE_TIMEOUT, CPU_MONITOR_INTERVAL, CPU_SAMPLE_DURATION = 30.0, 1.0, 0.07 # sec
    CPU_HIGH_THRESHOLD, CPU_CRITICAL_THRESHOLD = 85.0, 95.0 # %
    QUALITY_LEVELS = [
        {"sample_rate": 8000,  "put_accent": False, "put_yo": False, "name": "LOWEST"},
        {"sample_rate": 8000,  "put_accent": True,  "put_yo": False, "name": "LOW"},
        {"sample_rate": 24000, "put_accent": False, "put_yo": False, "name": "MEDIUM-LOW"},
        {"sample_rate": 24000, "put_accent": True,  "put_yo": True,  "name": "MEDIUM"},
        {"sample_rate": 48000, "put_accent": True,  "put_yo": False, "name": "HIGH"},
        {"sample_rate": 48000, "put_accent": True,  "put_yo": True,  "name": "MAXIMUM"}
    ]

class ModelLoader:
    """загрузка модели и инициализация torch"""
    @staticmethod
    def setup_torch(device):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if device.type == 'cpu':
            try: 
                cores = psutil.cpu_count(logical=False)
                if cores is None: cores = os.cpu_count()
            except: 
                cores = os.cpu_count()
            if cores is None: cores = 1 
            n_threads = 2 if (cores and cores > 1) else 1
            torch.set_num_threads(n_threads)
            torch.set_num_interop_threads(1)
            print(f"[INFO] CPU Threads: {n_threads} | Detected cores: {cores}")
        
        if device.type == 'cuda':
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    @staticmethod
    def download_model(model_path: str):
        if not os.path.exists(model_path):
            print(f"\n[ERROR] Model not found: {model_path}")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            try:
                import urllib.request
                print(f"[INFO] Downloading model...")
                urllib.request.urlretrieve(Config.URL, model_path, 
                    lambda b, bs, ts: print(f"\r {int(b*bs*100/ts)}%", end=''))
                print(f"\n[INFO] Model downloaded.")
            except Exception as e:
                print(f"\n[ERROR] Download failed: {e}")
                print(f"\n Please download manually from: {Config.URL}")
                input(f" and save as: {model_path}")
                sys.exit(1)
    
    @staticmethod
    def load_model(model_path: str, device):
        print(f"[INFO] Loading model '{model_path}'... ({device.type})")
        try:
            package = torch.package.PackageImporter(model_path)
            model = package.load_pickle("tts_models", "model")
            model.to(device)
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load model. File might be corrupted. Delete {model_path} and restart.")
            raise e

    @staticmethod
    def unload_model(model, device):
        if model is None: return
        del model
        
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("[INFO] CUDA cache cleared")
        
        print("[INFO] Model unloaded successfully")

class CPUMonitor:
    """мониторинг загрузки CPU"""
    def __init__(self):
        self.current_quality_level = len(Config.QUALITY_LEVELS) - 1
        self.current_load, self.last_change_time, self.max_level = 0.0, 0, len(Config.QUALITY_LEVELS) - 1
        self.lock, self.running, self.load_history, self.max_history_size = threading.Lock(), False, [], 3
        self.min_change_interval = 1.5
        self.monitor_thread = None
        self.last_activity_time = 0
    
    def start(self):
        with self.lock:
            if self.running: return
            self.last_activity_time = time.time()
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            if DEBUG: print("[DEBUG] CPU monitoring ON")
    
    def stop(self):
        with self.lock:
            if not self.running: return
            self.running = False
            if DEBUG: print("[DEBUG] CPU monitoring OFF")
    
    def record_activity(self):
        self.last_activity_time = time.time()
        if not self.running: self.start()
    
    def _check_idle_and_stop(self):
        if time.time() - self.last_activity_time >= Config.CPU_IDLE_TIMEOUT:
            self.stop()
            return True
        return False
    
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
    
    def get_current_quality_config(self) -> dict:
        with self.lock: return Config.QUALITY_LEVELS[self.current_quality_level].copy()
    
    def get_cpu_load(self) -> float:
        with self.lock: return self.current_load


class TextProcessor:
    """обработка текста (SSML, числа, транслитерация)"""
    pause0, pause1, pause2, pause3, pause4, pause5 = 0, 130, 180, 215, 320, 480
    BREAK_TIME_MAP = {'.': pause4, ',': pause2, '!': pause4, '?': pause4, 
                      '(': pause2, ')': pause2, '[': pause2, ']': pause2, 
                      ':': pause1, ';': pause3, '—': pause3, '…': pause5}
    EMOTIONS = {'!': (107, 0), '?': (93, 0)} # 'знак': (speed в %, pitch от -2 до 2)
    ALLOWED = frozenset("_~абвгдеёжзийклмнопрстуфхцчшщъыьэюя +.,!?…:;–")
    LATIN = frozenset("&%abcdefghijklmnopqrstuvwxyz")
    TRANSLIT_MAP = {'ough':'о','augh':'о','eigh':'эй','igh':'ай','tion':'шн','shch':'щ','ture': 'чер','sion': 'жн',
        'tch':'ч','sch':'ск','scr':'скр','thr':'тр','squ':'скв','ear':'ир','air':'эр','are':'эр','the':'зэ','and':'энд',
        'ea':'и','ee':'и','oo':'у','ai':'эй','ay':'эй','ei':'эй','ey':'эй','oi':'ой','oy':'ой','ou':'ау','ow':'ау','au':'о','aw':'о','ie':'и','ui':'у','ue':'ю','uo':'уо','eu':'ю','ew':'ю','oa':'о','oe':'о','sh':'ш','ch':'ч','zh':'ж','th':'з','kh':'х','ts':'ц','ph':'ф','wh':'в','gh':'г','qu':'кв','gu':'г','dg':'дж','ce':'це','ci':'си','cy':'си','ck':'к','ge':'дж','gi':'джи','gy':'джи','er':'эр',
        '&':'и', '%': ' процентов', # это замены, пока тут побудут
        'a':'а','b':'б','c':'к','d':'д','e':'е','f':'ф','g':'г','h':'х','i':'и','j':'дж','k':'к','l':'л','m':'м','n':'н','o':'о','p':'п','q':'к','r':'р','s':'с','t':'т','u':'у','v':'в','w':'в','x':'кс','y':'и','z':'з'}

    def __init__(self):
        self.base_speed = 1.0
        self.base_pitch = "medium"
        self.transl_trie = {}
        for k, v in self.TRANSLIT_MAP.items():
            node = self.transl_trie
            for ch in k:
                node = node.setdefault(ch, {})
            node['_'] = v

    def set_ssml_params(self, speed: float, pitch: str):
        self.base_speed, self.base_pitch = speed, pitch

    def _adj_pitch(self, pitch: str, delta: int) -> str:
        order = ["x-low", "low", "medium", "high", "x-high"]
        try:
            idx = max(0, min(len(order) - 1, order.index(pitch) + delta))
            return order[idx]
        except ValueError: return "medium"

    def process_text(self, text: str) -> str:
        if not text: return "", 0
        len_text = len(text)
        if len_text > Config.MAX_TEXT_LENGTH:
            len_text = Config.MAX_TEXT_LENGTH
            text = text[:len_text]
            print(f"[INFO] Text length truncated to {len_text} chars.")
        
        text = unquote(text).lower()
        has_latin = any(ch in self.LATIN for ch in text)
        processed_body = self._proc(text, len_text, has_latin)
        base_r = f"{int(self.base_speed * 100)}"
        base_p = self.base_pitch
        return f'<speak><prosody rate="{base_r}%" pitch="{base_p}">{processed_body}</prosody></speak>', len_text

    def _proc(self, text: str, len_text: int, has_latin: bool) -> str:
        res, buf, i, n = [], [], 0, len_text
        while i < n:
            ch = text[i]
            if ch.isdigit():
                i, p = self._num(text, i)
                buf.append(p)
                continue
            if has_latin and ch in self.LATIN:
                ni, tr = self._trans(text, i)
                if tr and tr != ch: buf.append(tr)
                i = ni
				# word_start = i
                # while i < n and text[i] in self.LATIN: i += 1
                # lat = text[word_start:i]
                # cyr = translit(lat, 'ru')
                # if cyr: buf.append(cyr)
                continue
            if ch in self.BREAK_TIME_MAP:
                skip = 1
                if ch == '.' and i + 2 < n and text[i+1] == '.':
                    if text[i+2] == '.':
                        ch = '…'
                        skip = 3
                s = ''.join(buf).strip()
                if s:
                    if ch in self.EMOTIONS: text_to_wrap = s + ' ' + ch
                    else: text_to_wrap = s
                    res.append(self._wrap(text_to_wrap, ch))
                res.append(f'<break time="{self.BREAK_TIME_MAP[ch]}ms"/>')
                buf.clear()
                i += skip
                continue
            if ch.isspace():
                if not buf or buf[-1] != ' ': buf.append(' ')
                i += 1
                continue
            if ch in self.ALLOWED:
                buf.append(ch)
                i += 1
                continue
            if not (buf and buf[-1] == ' '): buf.append(' ')
            i += 1
        if buf:
            s = ''.join(buf).strip()
            if s: res.append(self._wrap(s, None))
        return ''.join(res).strip()

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

    def _trans(self, text: str, pos: int) -> tuple:
        node, best, best_pos = self.transl_trie, None, pos
        j = pos
        while j < len(text) and text[j] in node:
            node = node[text[j]]; j += 1
            if '_' in node: best, best_pos = node['_'], j
        return (best_pos, best) if best else (pos + 1, text[pos] if text[pos] in self.ALLOWED else " ")
    
    def _wrap(self, txt: str, end_punct: str) -> str:
        if not txt: return ""
        if end_punct not in self.EMOTIONS: return txt
        
        sm, pd = self.EMOTIONS[end_punct]
        emo_r = f"{int(self.base_speed * sm)}"
        emo_p = self._adj_pitch(self.base_pitch, pd)
        words = txt.split()
        def attrs(rate, pitch): return f'rate="{rate}%" pitch="{pitch}"'
        
        if len(words) < 4:
            return f'<prosody {attrs(emo_r, emo_p)}>{txt}</prosody>'
        
        tail_count = max(1, int(len(words) * 0.2))
        
        if tail_count < len(words) and words[-1] in ['!', '?'] and tail_count == 1:
             tail_count = 2

        head_words = words[:-tail_count]
        tail_words = words[-tail_count:]
        head_text = " ".join(head_words)
        tail_text = " ".join(tail_words)
        return f'{head_text} <prosody {attrs(emo_r, emo_p)}>{tail_text}</prosody>'


@lru_cache(maxsize=512)
def num_to_words(num: str) -> str:
    """преобразование чисел в слова"""
    if not num or not num.isdigit() or len(num) > 9: return str(num) if num else ""
    return num2words(int(num), lang='ru')


class AudioSynthesizer:
    """генерация звука (синтез речи из SSML)"""
    def __init__(self, model, device, cpu_monitor):
        self.model = model
        self.device = device
        self.cpu_monitor = cpu_monitor
        self.inference_count = 0
        self.clean_cuda_every = 50 

    def _to_wav(self, t, sr):
        d = t.detach().cpu().numpy().squeeze()
        raw = np.clip(d * 32767, -32768, 32767).astype(np.int16).tobytes()
        del d
        sz = len(raw)
        hdr = b'RIFF' + struct.pack('<I', 36 + sz) + b'WAVEfmt '
        hdr += struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16)
        hdr += b'data' + struct.pack('<I', sz)
        return hdr + raw

    def synthesize(self, ssml: str, speaker_name: str, sample_rate: int, put_accent: bool, put_yo: bool, vol_boost: float) -> bytes:
        try:
            with torch.no_grad():
                if DEBUG and self.device.type == 'cuda':
                    torch.cuda.synchronize()
            
                audio = self.model.apply_tts(
                    ssml_text=ssml, speaker=speaker_name, sample_rate=sample_rate, 
                    put_accent=put_accent, put_yo=put_yo)
            
                if self.device.type == 'cuda':
                    self.inference_count += 1
                    if self.inference_count >= self.clean_cuda_every:
                        torch.cuda.empty_cache()
                        self.inference_count = 0

        except Exception as e:
            raise RuntimeError(f"Model inference failed: {str(e)}")
        
        if audio.dim() == 1: 
            audio = audio.unsqueeze(0)
        if vol_boost != 0:
            audio = torch.clamp(audio * (10 ** (vol_boost / 20.0)), -1.0, 1.0)

        wav_bytes = self._to_wav(audio, sample_rate)
        del audio
        return wav_bytes


class TTSService:
    """координация"""
    def __init__(self, model, device, cpu_monitor):
        self.text_processor = TextProcessor()
        self.audio_synthesizer = AudioSynthesizer(model, device, cpu_monitor)
        self.cpu_monitor = cpu_monitor
    
    def synthesize_speech(self, text: str, speaker_id: int, length: float, pitch_adj: float) -> bytes:
        t_start = time.time() if DEBUG else None
        
        if not (0 <= speaker_id < len(SPEAKERS)): 
            raise ValueError(f"Bad Speaker ID: {speaker_id}")
        
        self.cpu_monitor.record_activity()
        q = self.cpu_monitor.get_current_quality_config()
        sr = min(Config.SAMPLE_RATE, q["sample_rate"])
        
        spk = SPEAKERS[speaker_id]
        speed = max(0.1, min(10.0, spk['base_speed'] * (2 ** ((1 - length) * (5 if length >= 1 else 15) / 10))))
        
        p_map = {"x-low": -10, "low": -4, "medium": 0, "high": 4, "x-high": 10}
        base_p = p_map.get(spk['pitch'], 0)
        final_p = max(-10, min(10, base_p + pitch_adj))
        p_str = next((k for k, v in p_map.items() if v == final_p), "medium")
        
        self.text_processor.set_ssml_params(speed, p_str)
        ssml, len_text = self.text_processor.process_text(text)
        
        if DEBUG: 
            print(f"[DEBUG] Quality: {q['name']} Spk:{spk['name']} Speed:{speed:.2f} Pitch:{p_str} SR:{sr}")
            print(f"[DEBUG] Text len:{len_text} SSML:{ssml}")
        
        wav_bytes = self.audio_synthesizer.synthesize(
            ssml, spk['name'], sr, 
            q['put_accent'], q['put_yo'], 
            spk['vol_boost']
        )
        
        if DEBUG and t_start:
            num_samples = len(wav_bytes) - 44
            dur = num_samples / (sr * 2)
            if dur > 0: 
                print(f"[DEBUG] Time:{(time.time()-t_start)*1000:.0f}ms Dur:{dur:.2f}s RTF:{(time.time()-t_start)/dur:.2f}")
        
        return wav_bytes


class HTTPServer:
    """HTTP сервер"""
    def __init__(self, tts_service):
        self.app = Bottle()
        self.tts_service = tts_service
        self._setup_routes()
        self._setup_cors()
    
    def _setup_cors(self):
        @self.app.hook('after_request')
        def enable_cors():
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With'
            response.headers['Access-Control-Max-Age'] = '86400'
        
        @self.app.route('/voice/speakers', method='OPTIONS')
        @self.app.route('/voice/vits', method='OPTIONS')
        def options_handler():
            response.status = 200
            return ''
    
    def _setup_routes(self):
        @self.app.route('/voice/speakers', method='GET')
        def get_speakers():
            response.content_type = 'application/json'
            print("[HTTP] Request list of speakers from LunaTranslator.")
            return {"vits": [{"id": s["id"], "name": s["name"], "lang": s["lang"]} for s in SPEAKERS]}
        
        @self.app.route('/voice/vits', method='GET')
        def synthesize():
            text, speaker_id = request.query.text or "", int(request.query.id or 0)
            length, pitch = float(request.query.length or 1.0), float(request.query.pitch or 0)
            
            if not text: 
                response.status = 400
                return {"error": "Text is required"}
            
            print(f"[HTTP] New request: ID={speaker_id}")
            
            try:
                audio_data = self.tts_service.synthesize_speech(text, speaker_id, length, pitch)
                response.content_type = 'audio/wav'
                response.headers['Content-Length'] = str(len(audio_data))
                return audio_data
            except Exception as e:
                print(f"[ERROR] Synthesis failed: {e}")
                response.status = 500
                response.content_type = 'text/plain'
                return str(e)
    
    def run(self, host: str, port: int):
        run(self.app, host=host, port=port, quiet=True, server='wsgiref')


class Application:
    """запуск приложения"""
    
    def __init__(self):
        self.model = None
        self.cpu_monitor = None
        self.tts_service = None
        self.http_server = None
        self.running = False
    
    def initialize(self):
        ModelLoader.download_model(Config.MODEL_PATH)
        ModelLoader.setup_torch(Config.DEVICE)
        
        self.model = ModelLoader.load_model(Config.MODEL_PATH, Config.DEVICE)
        self.cpu_monitor = CPUMonitor()
        self.tts_service = TTSService(self.model, Config.DEVICE, self.cpu_monitor)
        self.http_server = HTTPServer(self.tts_service)
    
    def stop(self, signum=None, frame=None):
        if not self.running: return
        self.running = False
        
        if self.cpu_monitor:
            self.cpu_monitor.stop()
        
        if self.model:
            try: ModelLoader.unload_model(self.model, Config.DEVICE)
            except: pass
            self.model = None
        
        num_to_words.cache_clear()
        
        def exit_now(): os._exit(0)
        threading.Timer(1, exit_now).start()
        print("[INFO] Application stopped successfully")
    
    def _win_handler(self, dwCtrlType):
        if dwCtrlType in [0, 2]:
            self.stop()
            return True
        return False
        
    def run(self):
        self.initialize()
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        
        if sys.platform == "win32":
            kernel32 = ctypes.windll.kernel32
            HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)(self._win_handler)
            kernel32.SetConsoleCtrlHandler(HandlerRoutine, True)
            
        print('=' * 60)
        print(f" Silero TTS Server for LunaTranslator v{MAIN_VERSION}")
        print(f" Device: {str(Config.DEVICE).upper()} | http://{Config.HOST}:{Config.PORT}")
        if DEBUG: print(" DEBUG mode ON")
        print('=' * 60)
        self.running = True
        
        try: self.http_server.run(Config.HOST, Config.PORT)
        except Exception as e: print(f"[ERROR] Server error: {e}")
        finally: self.stop()

if __name__ == "__main__":
    Application().run()