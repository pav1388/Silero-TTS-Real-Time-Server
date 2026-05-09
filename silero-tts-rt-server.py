# silero-tts-rt-server.py
# pav13

import ctypes, logging, os, platform, signal, struct, sys, threading, time
import numpy as np
import torch
from bottle import Bottle, hook, request, response, run
from functools import lru_cache
from num2words import num2words
from urllib.parse import unquote

import warnings
warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype")

MAIN_VERSION = "0.8.0"
DEBUG = ('--debug' in sys.argv) or (os.environ.get('DEBUG', '0').lower() in ('1', 'true'))
CUDA = ('--cuda' in sys.argv or '--gpu' in sys.argv) or (os.environ.get('CUDA', '0').lower() in ('1', 'true'))
NO_CPU_MONITOR = ('--no-cpu-monitor' in sys.argv) or (os.environ.get('NO_CPU_MONITOR', '0').lower() in ('1', 'true'))

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if not NO_CPU_MONITOR:
    try:
        from cpu_monitor import CPUMonitor
        HAS_CPU_MONITOR = True
    except ImportError:
        HAS_CPU_MONITOR = False
        CPUMonitor = None
        logger.warning("cpu_monitor.py not found. Speech quality always 'MAX'.")
else:
    HAS_CPU_MONITOR = False
    CPUMonitor = None
    logger.debug("CPU Monitor disabled. Speech quality always 'MAX'.")


class Config:
    """Конфигурация приложения"""
    if CUDA:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device('cpu')
    MODEL_PATH = "models/v5_5_ru.pt"
    MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
    HOST, PORT = "127.0.0.1", 23457
    MAX_SAMPLE_RATE = 48000  # 48000, 24000, 8000
    MAX_TEXT_LENGTH = 900
    PITCH_ORDER = ["x-low", "low", "medium", "high", "x-high"]
    MAX_QUALITY_CONFIG = {"sample_rate": 48000, "put_accent": True,  "put_yo": True,
            "put_stress_homo": True,  "put_yo_homo": True,  "name": "MAX"}
    SPEAKERS = []
    REAL_SPEAKERS_COUNT = 0
    SPEAKERS_INFO = {"aidar": {"gender": "male"}, "baya": {"gender": "female"},
        "kseniya": {"gender": "female"}, "eugene": {"gender": "male"}, "xenia": {"gender": "female"}}
    

class ModelLoader:
    """загрузка модели и инициализация torch"""
    @staticmethod
    def setup_torch(device):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        if device.type == 'cpu':
            cores = os.cpu_count() or 1
            n_threads = 2 if cores > 1 else 1
            torch.set_num_threads(n_threads)
            torch.set_num_interop_threads(1)
            logger.info(f"Device: CPU | Threads: {n_threads} | Logical cores: {cores}")
        
        elif device.type == 'cuda':
            logger.info(f"Device: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            
            if DEBUG:
                logger.debug(f"CUDA: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")
                props = torch.cuda.get_device_properties(0)
                logger.debug(f"Capability: {props.major}.{props.minor} | SMs: {props.multi_processor_count}")
        
        if DEBUG:
            try:
                system = platform.system()
                if system == "Windows":
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]
                    memoryStatus = MEMORYSTATUSEX()
                    memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
                    total_mem = memoryStatus.ullTotalPhys
                    avail_mem = memoryStatus.ullAvailPhys
                else:
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.readlines()
                    total_mem = int(meminfo[0].split()[1]) * 1024
                    avail_mem = int(meminfo[2].split()[1]) * 1024
                
                logger.debug(f"RAM: {total_mem/1e9:.1f} GB | Free: {avail_mem/1e9:.1f} GB")
                logger.debug(f"PyTorch: {torch.__version__} | Python: {sys.version.split()[0]}")
                logger.debug(f"OS: {platform.system()} {platform.release()} | Arch: {platform.machine()}")
            except Exception as e:
                if DEBUG:
                    logger.warning(f"Could not get system memory info: {e}")
    
    @staticmethod
    def download_model(model_path: str):
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            try:
                import urllib.request
                logger.info("Downloading model (~140Mb)...")
                urllib.request.urlretrieve(Config.MODEL_URL, model_path, 
                    lambda b, bs, ts: print(f"\r {int(b*bs*100/ts)}%", end=''))
                logger.info("Model downloaded.")
            except Exception as e:
                logger.error(f"Download failed: {e}")
                logger.error(f"Please download manually from: {Config.MODEL_URL}")
                input(f" and save as: {model_path}")
                sys.exit(1)
    
    @staticmethod
    def load_model(model_path: str, device):
        logger.info(f"Loading model '{model_path}'...")
        try:
            model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
            model.to(device)
            logger.info("OK")
            model_speakers = model.speakers
            
            Config.SPEAKERS = [
                {"id": idx, "name": name, 
                 "gender": Config.SPEAKERS_INFO.get(name, {}).get("gender", "n/d"),
                 "lang": "ru"}
                for idx, name in enumerate(model_speakers)
            ]
            Config.REAL_SPEAKERS_COUNT = len(model_speakers)
            
            special_speakers = [
                {"id": Config.REAL_SPEAKERS_COUNT,     "name": "RANDOM",    "gender": "both",   "lang": "ru"},
                {"id": Config.REAL_SPEAKERS_COUNT + 1, "name": "RANDOM_M",  "gender": "male",   "lang": "ru"},
                {"id": Config.REAL_SPEAKERS_COUNT + 2, "name": "RANDOM_F",  "gender": "female", "lang": "ru"},
                {"id": Config.REAL_SPEAKERS_COUNT + 3, "name": "HASH",      "gender": "both",   "lang": "ru"},
            ]
            Config.SPEAKERS.extend(special_speakers)
            
            if DEBUG:
                logger.debug(f"Loaded {Config.REAL_SPEAKERS_COUNT} speakers from model:")
                for spk in Config.SPEAKERS[:Config.REAL_SPEAKERS_COUNT]:
                    logger.debug(f"  - {spk['name']}: {spk['lang']}, {spk['gender']};")
            
            return model
        except Exception as e:
            logger.error("FAIL")
            logger.error(f"Failed to load model. File might be corrupted. Delete {model_path} and restart.")
            raise e

    @staticmethod
    def unload_model(model, device):
        if model is None: return
        logger.info("Unloading model...")
        try:
            del model
            if device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            logger.info("OK")
        except Exception as e:
            logger.error("FAIL")
            raise e


class TextProcessor:
    """обработка текста (SSML, числа, транслитерация, разбиение на предложения)"""
    pause0, pause1, pause2, pause3, pause4, pause5 = 0, 130, 180, 215, 320, 480
    PUNCT_REPL = {'.': pause4, ',': pause2, '(': pause2, ')': pause2, '[': pause2, ']': pause2, 
                    ':': pause1, ';': pause3, '—': pause3, '…': pause5}
    PUNCT_NO_REPL = {'!': pause4, '?': pause4}
    ALLOWED = frozenset("_~абвгдеёжзийклмнопрстуфхцчшщъыьэюя +.,!?…:;–*")
    LATIN = frozenset("abcdefghijklmnopqrstuvwxyz&")
    TRANSLIT_MAP = {'ough':'о','augh':'о','eigh':'эй','igh':'ай','tion':'шн','shch':'щ','ture': 'чер','sion': 'жн',
        'tch':'ч','sch':'ск','scr':'скр','thr':'тр','squ':'скв','ear':'ир','air':'эр','are':'эр','the':'зэ','and':'энд',
        'ea':'и','ee':'и','oo':'у','ai':'эй','ay':'эй','ei':'эй','ey':'эй','oi':'ой','oy':'ой','ou':'ау','ow':'ау','au':'о','aw':'о','ie':'и','ui':'у','ue':'ю','uo':'уо','eu':'ю','ew':'ю','oa':'о','oe':'о','sh':'ш','ch':'ч','zh':'ж','th':'з','kh':'х','ts':'ц','ph':'ф','wh':'в','gh':'г','qu':'кв','gu':'г','dg':'дж','ce':'це','ci':'си','cy':'си','ck':'к','ge':'дж','gi':'джи','gy':'джи','er':'эр',
        'a':'а','b':'б','c':'к','d':'д','e':'е','f':'ф','g':'г','h':'х','i':'и','j':'дж','k':'к','l':'л','m':'м','n':'н','o':'о','p':'п','q':'к','r':'р','s':'с','t':'т','u':'у','v':'в','w':'в','x':'кс','y':'и','z':'з','&':'и'}

    def __init__(self):
        self.speed_percent = 100
        self.pitch_level = "medium"
        self.transl_trie = {}
        for k, v in self.TRANSLIT_MAP.items():
            node = self.transl_trie
            for ch in k:
                node = node.setdefault(ch, {})
            node['_'] = v

    def set_ssml_params(self, speed_percent: int, pitch: str):
        self.speed_percent = max(40, min(300, speed_percent))
        self.pitch_level = pitch if pitch in Config.PITCH_ORDER else "medium"

    def split_sentences(self, text: str) -> list:
        sentences = []
        current_sentence = []
        end_of_sentence_chars = {'.', '!', '?', '…'}
        len_text = len(text)
        
        for i, ch in enumerate(text):
            current_sentence.append(ch)
            if ch in end_of_sentence_chars:
                if i + 1 == len_text or text[i + 1] == ' ':
                    sentences.append(''.join(current_sentence).strip())
                    current_sentence = []
        
        if current_sentence:
            sentences.append(''.join(current_sentence).strip())
        
        sentences = [s for s in sentences if s]
        result = []
        for sentence in sentences:
            if len(sentence) <= 200:
                result.append(sentence)
                continue

            parts = []
            current_part = []
            minor_delim_chars = {',', ';', ':', '—'}
            
            for ch in sentence:
                current_part.append(ch)
                if ch in minor_delim_chars:
                    parts.append(''.join(current_part).strip())
                    current_part = []
            if current_part:
                parts.append(''.join(current_part).strip())
            parts = [p for p in parts if p]
            current = []
            cur_len = 0
            cur_join = " ".join

            for part in parts:
                if cur_len and cur_len < 60:
                    current.append(part)
                    cur_len += len(part) + 1
                elif current:
                    result.append(cur_join(current))
                    current = [part]
                    cur_len = len(part)
                else:
                    current = [part]
                    cur_len = len(part)

            if current:
                if result and cur_len < 60:
                    result[-1] += " " + cur_join(current)
                else:
                    result.append(cur_join(current))
        
        return result

    def process_sentence(self, text: str, vol_boost: float) -> tuple:
        len_text = len(text)
        if len_text > Config.MAX_TEXT_LENGTH:
            len_text = Config.MAX_TEXT_LENGTH
            text = text[:len_text]
            logger.info(f"Text length truncated to {len_text} chars.")
        
        text = unquote(text).lower()
        has_latin = any(ch in self.LATIN for ch in text)
        vol_boost_mod = vol_boost + 3.5 if text.rstrip().endswith('!') else vol_boost
        processed_text = self._proc(text, len_text, has_latin)
        ssml = f'<speak><prosody rate="{self.speed_percent}%" pitch="{self.pitch_level}">{processed_text}</prosody></speak>'
        
        return ssml, vol_boost_mod

    def _proc(self, text: str, len_text: int, has_latin: bool) -> str:
        res, buf, i = [], [], 0
        PUNCT_REPL, PUNCT_NO_REPL = self.PUNCT_REPL, self.PUNCT_NO_REPL
        ALLOWED, LATIN = self.ALLOWED, self.LATIN
        while i < len_text:
            ch = text[i]
            if ch.isdigit():
                i, p = self._num(text, i)
                buf.append(p)
                continue
            if has_latin and ch in LATIN:
                ni, tr = self._trans(text, i)
                if tr and tr != ch: buf.append(tr)
                i = ni
                continue
            if ch in PUNCT_REPL:
                skip = 1
                if ch == '.' and i + 2 < len_text and text[i+1] == '.' and text[i+2] == '.':
                    ch = '…'
                    skip = 3
                if buf:
                    s = ''.join(buf).strip()
                    if s:
                        res.append(s)
                res.append(f'<break time="{PUNCT_REPL[ch]}ms"/>')
                buf.clear()
                i += skip
                continue
            if ch in PUNCT_NO_REPL:
                if buf:
                    s = ''.join(buf).strip()
                    if s:
                        res.append(self._wrap(s))
                    res.append(ch) 
                res.append(f'<break time="{PUNCT_NO_REPL[ch]}ms"/>')
                buf.clear()
                i += 1
                continue
            if ch.isspace():
                if not buf or buf[-1] != ' ': buf.append(' ')
                i += 1
                continue
            if ch in ALLOWED:
                buf.append(ch)
                i += 1
                continue
            if not (buf and buf[-1] == ' '): buf.append(' ')
            i += 1
        
        if buf:
            s = ''.join(buf).strip()
            if s: 
                res.append(s)
        return ''.join(res).strip()
    
    def _trans(self, text: str, pos: int) -> tuple:
        node, best, best_pos = self.transl_trie, None, pos
        j = pos
        n = len(text)
        while j < n and text[j] in node:
            node = node[text[j]]; j += 1
            if '_' in node: best, best_pos = node['_'], j
        return (best_pos, best) if best else (pos + 1, text[pos] if text[pos] in self.ALLOWED else " ")
    
    def _wrap(self, text: str) -> str:
        last_space = text.rfind(' ')
        if last_space == -1:
            return f"*{text}*"
        
        return f"{text[:last_space]} *{text[last_space+1:]}*"

    def _num(self, text: str, start: int) -> tuple:
        i, n = start, len(text)
        while i < n and text[i].isdigit(): i += 1
        if i == start: return start + 1, text[start]

        num_str = text[start:i]
        num_val = int(num_str)

        # Время
        if i < n and text[i] == ':' and i + 2 < n and text[i+1:i+3].isdigit():
            mm = int(text[i+1:i+3])
            end = i + 3
            res = f"{num_to_words(num_val)} часов {num_to_words(mm)} минут"
            if end < n and text[end] == ':' and end + 2 < n and text[end+1:end+3].isdigit():
                res += f" {num_to_words(int(text[end+1:end+3]))} секунд"
                end += 3
            return end, res

        # Дата
        if i < n and text[i] in '.-/' and i + 1 < n and text[i+1].isdigit():
            j = i + 1
            while j < n and text[j].isdigit(): j += 1
            mm = int(text[i+1:j])

            if j < n and text[j] in '.-/' and j + 1 < n and text[j+1].isdigit():
                l = j + 1
                while l < n and text[l].isdigit(): l += 1
                part3 = int(text[j+1:l])
                months = ["", "января", "февраля", "марта", "апреля", "мая", "июня",
                          "июля", "августа", "сентября", "октября", "ноября", "декабря"]
                month_w = months[mm] if 1 <= mm <= 12 else str(mm)

                if (i - start) == 4:
                    year_w = num_to_words(num_val, to='ordinal', case='g')
                    day_w = num_to_words(part3, to='ordinal', gender='n')
                else:
                    day_w = num_to_words(num_val, to='ordinal', gender='n')
                    year_w = num_to_words(part3, to='ordinal', case='g')

                return l, f"{day_w} {month_w} {year_w} года"

        # Десятичные дроби
        if i < n and text[i] in '.,' and i + 1 < n and text[i+1].isdigit():
            k = i + 1
            while k < n and text[k].isdigit(): k += 1
            if num_val == 1 and text[i+1:k] == '5':
                return k, "полтора"
            return k, num_to_words(text[start:k].replace(',', '.'))

        # Проценты
        if i < n and text[i] == '%':
            return i + 1, f"{num_to_words(num_val)} {self._plur(num_val, ('процент', 'процента', 'процентов'))}"

        # Обычные дроби
        if i < n and text[i] == '/' and i + 1 < n and text[i+1].isdigit():
            k = i + 1
            while k < n and text[k].isdigit(): k += 1
            return k, f"{num_to_words(num_val)} дробь {num_to_words(int(text[i+1:k]))}"

        # Обычное число
        return i, num_to_words(num_val)

    @staticmethod
    def _plur(n: int, forms: tuple) -> str:
        if n % 100 in (11, 12, 13, 14): return forms[2]
        if n % 10 == 1: return forms[0]
        if n % 10 in (2, 3, 4): return forms[1]
        return forms[2]


@lru_cache(maxsize=4096)
def num_to_words(num, to='cardinal', case='n', gender='m', plural=False, animate=False) -> str:
    """num: int/str | to: cardinal/ordinal | case: n/g/d/a/i/p (им/род/дат/вин/твор/предл)
       gender: m/f/n | plural: bool | animate: bool (одуш., только для вин. п.)"""
    return num2words(num, lang='ru', to=to, case=case, gender=gender, plural=plural, animate=animate)


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

    def synthesize(self, ssml: str, speaker_name: str, sample_rate: int, 
                    put_accent: bool, put_yo: bool, put_stress_homo: bool, put_yo_homo: bool,
                        vol_boost: float) -> tuple:
        t_start = time.time() if DEBUG else None
        audio = None
        try:
            with torch.no_grad():
                if DEBUG and self.device.type == 'cuda':
                    torch.cuda.synchronize()
            
                audio = self.model.apply_tts(
                    ssml_text=ssml, speaker=speaker_name, sample_rate=sample_rate, 
                    put_accent=put_accent, put_yo=put_yo,
                    put_stress_homo=put_stress_homo, put_yo_homo=put_yo_homo, intensity=3)
            
                if self.device.type == 'cuda':
                    self.inference_count += 1
                    if self.inference_count >= self.clean_cuda_every:
                        torch.cuda.empty_cache()
                        self.inference_count = 0

        except Exception as e:
            if audio is not None: del audio
            raise RuntimeError(f"Model inference failed: {str(e)}")
        
        if audio.dim() == 1: 
            audio = audio.unsqueeze(0)
        if vol_boost != 0:
            audio = torch.clamp(audio * (10 ** (vol_boost / 20.0)), -1.0, 1.0)

        wav_bytes = self._to_wav(audio, sample_rate)
        
        duration = 0
        inference_time = 0
        if DEBUG:
            num_samples = audio.shape[1] if audio.dim() > 1 else audio.shape[0]
            duration = num_samples / sample_rate
            inference_time = time.time() - t_start
        del audio
        return wav_bytes, duration, inference_time


class TTSService:
    """координация"""
    def __init__(self, model, device, cpu_monitor):
        self.text_processor = TextProcessor()
        self.audio_synthesizer = AudioSynthesizer(model, device, cpu_monitor)
        self.cpu_monitor = cpu_monitor
    
    def _resolve_speaker(self, speaker_id: int, text: str = "") -> tuple:
        real_speakers_count = Config.REAL_SPEAKERS_COUNT

        if speaker_id == real_speakers_count: # RANDOM (оба пола)
            speaker_id = int(time.time() * 100000) % real_speakers_count
        elif speaker_id == real_speakers_count + 1: # RANDOM_M (только мужские)
            male_speakers = [i for i, s in enumerate(Config.SPEAKERS[:real_speakers_count]) if s.get('gender') == 'male']
            if male_speakers:
                speaker_id = male_speakers[int(time.time() * 100000) % len(male_speakers)]
            else:
                speaker_id = int(time.time() * 100000) % real_speakers_count
        elif speaker_id == real_speakers_count + 2: # RANDOM_F (только женские)
            female_speakers = [i for i, s in enumerate(Config.SPEAKERS[:real_speakers_count]) if s.get('gender') == 'female']
            if female_speakers:
                speaker_id = female_speakers[int(time.time() * 100000) % len(female_speakers)]
            else:
                speaker_id = int(time.time() * 100000) % real_speakers_count
        elif speaker_id == real_speakers_count + 3: # HASH (на основе текста)
            t = text[:500] if text else str(time.time())
            h = 5381
            for c in t:
                h = ((h << 5) + h) ^ ord(c)
            speaker_id = (h & 0x7fffffff) % real_speakers_count
        
        if not (0 <= speaker_id < real_speakers_count):
            speaker_id = 0
        
        return speaker_id, Config.SPEAKERS[speaker_id]
    
    def speakers_list(self):
        return {"silero": Config.SPEAKERS.copy()}
    
    def _get_quality_config(self):
        if self.cpu_monitor:
            self.cpu_monitor.record_activity()
            if DEBUG: logger.debug(self.cpu_monitor.get_status())
            return self.cpu_monitor.get_current_quality_config()
        return Config.MAX_QUALITY_CONFIG
    
    def _synthesize_sentence(self, sentence: str, speaker_name: str, 
                             speed: int, pitch: str, vol_boost: float, quality_config: dict) -> tuple:
        self.text_processor.set_ssml_params(speed, pitch)
        ssml, vol_boost_mod = self.text_processor.process_sentence(sentence, vol_boost)
        
        sr = min(Config.MAX_SAMPLE_RATE, quality_config["sample_rate"])
        
        if DEBUG:
            logger.debug(f"  N:{speaker_name} S:{speed}% P:{pitch} V_B:{vol_boost_mod}dB S_R:{sr} Q:{quality_config['name']}")
            logger.debug(f"  SSML_L:{len(ssml)} SSML:{ssml}")
        
        return self.audio_synthesizer.synthesize(
            ssml, speaker_name, sr,
            quality_config['put_accent'], quality_config['put_yo'],
            quality_config['put_stress_homo'], quality_config['put_yo_homo'],
            vol_boost_mod
        )
    
    def synthesize_stream(self, text, speaker_id, speed, pitch, vol_boost, r_count):
        resolved_speaker_id, speaker = self._resolve_speaker(speaker_id, text)
        speaker_name = speaker['name']
        sentences = self.text_processor.split_sentences(text)
        
        if not sentences:
            raise ValueError("No valid sentences found")
        
        logger.info(f"Stream request #{r_count}: {len(sentences)} sentences.")
        
        if DEBUG:
            for i, s in enumerate(sentences):
                logger.debug(f"  [{i+1}] {s[:80]}{'...' if len(s) > 80 else ''}")
        
        total_duration = 0
        total_time = 0
        
        def generate():
            nonlocal total_duration, total_time
            first_chunk = True
            
            for i, sentence in enumerate(sentences):
                try:
                    if DEBUG:
                        logger.debug(f">>Processing sentence {i+1}/{len(sentences)}")
                    
                    quality_config = self._get_quality_config()
                    wav_bytes, duration, inference_time = self._synthesize_sentence(
                        sentence, speaker_name, speed, pitch, vol_boost, quality_config
                    )
                    if DEBUG and duration > 0:
                        total_duration += duration
                        total_time += inference_time
                        logger.debug(f"  Duration: {duration:.2f}s | Time: {inference_time*1000:.0f}ms | RTF: {inference_time/duration:.3f}")
                    
                    if len(wav_bytes) > 44 and wav_bytes[:4] == b'RIFF':
                        offset = 12
                        while offset < len(wav_bytes) - 8:
                            chunk_id = wav_bytes[offset:offset+4]
                            chunk_size = struct.unpack('<I', wav_bytes[offset+4:offset+8])[0]

                            if chunk_id == b'data':
                                if first_chunk:
                                    header = bytearray(wav_bytes[:offset+8])
                                    struct.pack_into('<I', header, 4, 0xFFFFFFFF)
                                    struct.pack_into('<I', header, offset+4, 0xFFFFFFFF)
                                    yield bytes(header)
                                    first_chunk = False

                                yield wav_bytes[offset+8:offset+8+chunk_size]
                                break

                            offset += 8 + chunk_size
                    else:
                        yield wav_bytes
                    
                except Exception as e:
                    logger.error(f"Error processing sentence {i+1}: {e}")
                    continue
            
            if DEBUG and total_duration > 0:
                logger.debug(f"  Stream #{r_count} completed. Total duration: {total_duration:.2f}s | Total time: {total_time*1000:.0f}ms | Avg RTF: {total_time/total_duration:.3f}")
        
        return generate()
    
    def synthesize_once(self, text, speaker_id, speed, pitch, vol_boost, r_count):
        resolved_speaker_id, speaker = self._resolve_speaker(speaker_id, text)
        speaker_name = speaker['name']
        sentences = self.text_processor.split_sentences(text)
        
        if not sentences:
            raise ValueError("No valid sentences found")
        
        logger.info(f"Speech request #{r_count}: {len(sentences)} sentences.")
        
        if DEBUG:
            for i, s in enumerate(sentences):
                logger.debug(f"  [{i+1}] {s[:80]}{'...' if len(s) > 80 else ''}")
        
        all_audio_data = bytearray()
        first_sentence = True
        total_duration = 0
        total_time = 0
        
        for i, sentence in enumerate(sentences):
            try:
                if DEBUG:
                    logger.debug(f">>Processing sentence {i+1}/{len(sentences)}")
                    
                
                quality_config = self._get_quality_config()
                wav_bytes, duration, inference_time = self._synthesize_sentence(
                    sentence, speaker_name, speed, pitch, vol_boost, quality_config
                )
                if DEBUG and duration > 0:
                    total_duration += duration
                    total_time += inference_time
                    logger.debug(f"  Duration: {duration:.2f}s | Time: {inference_time*1000:.0f}ms | RTF: {inference_time/duration:.3f}")
                
                if first_sentence:
                    all_audio_data.extend(wav_bytes)
                    first_sentence = False
                else:
                    if len(wav_bytes) > 44:
                        all_audio_data.extend(wav_bytes[44:])
                    else:
                        all_audio_data.extend(wav_bytes)
                    
            except Exception as e:
                logger.error(f"Error processing sentence {i+1}: {e}")
                continue
        
        if not all_audio_data:
            raise RuntimeError("No audio data generated")
        
        if len(all_audio_data) > 44 and all_audio_data[:4] == b'RIFF':
            total_size = len(all_audio_data) - 8
            struct.pack_into('<I', all_audio_data, 4, total_size)
            
            offset = 12
            while offset < len(all_audio_data) - 8:
                chunk_id = all_audio_data[offset:offset+4]
                if chunk_id == b'data':
                    data_size = len(all_audio_data) - offset - 8
                    struct.pack_into('<I', all_audio_data, offset+4, data_size)
                    break
                chunk_size = struct.unpack('<I', all_audio_data[offset+4:offset+8])[0]
                offset += 8 + chunk_size
        
        if DEBUG and total_duration > 0:
            logger.debug(f"  Speech #{r_count} completed. Total duration: {total_duration:.2f}s | Total time: {total_time*1000:.0f}ms | Avg RTF: {total_time/total_duration:.3f}")
        
        return bytes(all_audio_data)


class HTTPServer:
    """HTTP сервер"""
    def __init__(self, tts_service, application):
        self.app = Bottle()
        self.tts_service = tts_service
        self.application = application
        self._setup_routes()
        self._setup_cors()
        self.r_count = 0
    
    def _setup_cors(self):
        @self.app.hook('after_request')
        def enable_cors():
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With'
            response.headers['Access-Control-Max-Age'] = '86400'
        
        @self.app.route('/speakers', method='OPTIONS')
        @self.app.route('/speak', method='OPTIONS')
        @self.app.route('/speak/stream', method='OPTIONS')
        @self.app.route('/restart', method='OPTIONS')
        def options_handler():
            response.status = 200
            return ''
    
    def _setup_routes(self):
        @self.app.route('/speakers', method='GET')
        def speakers():
            response.content_type = 'application/json'
            logger.info("Request list of speakers.")
            return self.tts_service.speakers_list()
        
        @self.app.route('/speak', method='GET')
        def speak():
            text = request.query.text or ""
            speaker_id = int(request.query.id or 0)
            speed = int(request.query.speed or 100)
            pitch = request.query.pitch or "medium"
            vol_boost = float(request.query.vol_boost or 0)
            
            if not text: 
                response.status = 400
                return {"error": "Text is required"}
            
            self.r_count += 1

            try:
                audio_data = self.tts_service.synthesize_once(
                    text, speaker_id, speed, pitch, vol_boost, self.r_count
                )
                response.content_type = 'audio/wav'
                response.headers['Content-Length'] = str(len(audio_data))
                return audio_data
                    
            except ValueError as ve:
                logger.warning(f"Validation error: {ve}")
                response.status = 400
                return {"error": str(ve)}
            except Exception as e:
                logger.error(f"Synthesis failed: {e}")
                response.status = 500
                response.content_type = 'text/plain'
                return str(e)
        
        @self.app.route('/speak/stream', method='GET')
        def speak_stream():
            text = request.query.text or ""
            speaker_id = int(request.query.id or 0)
            speed = int(request.query.speed or 100)
            pitch = request.query.pitch or "medium"
            vol_boost = float(request.query.vol_boost or 0)
            
            if not text: 
                response.status = 400
                return {"error": "Text is required"}
            
            self.r_count += 1

            try:
                gen = self.tts_service.synthesize_stream(
                    text, speaker_id, speed, pitch, vol_boost, self.r_count
                )
                response.content_type = 'application/octet-stream'
                response.headers['Cache-Control'] = 'no-cache'
                response.headers['X-Accel-Buffering'] = 'no'
                return gen
                    
            except ValueError as ve:
                logger.warning(f"Validation error: {ve}")
                response.status = 400
                return {"error": str(ve)}
            except Exception as e:
                logger.error(f"Synthesis failed: {e}")
                response.status = 500
                response.content_type = 'text/plain'
                return str(e)
        
        @self.app.route('/restart', method='POST')
        def restart():
            try:
                logger.info("Reboot request.")
                threading.Thread(target=self.application.restart, daemon=True).start()
                response.status = 200
                return {"status": "success", "message": "Restarting..."}
            except Exception as e:
                logger.error(f"Restart failed: {e}")
                response.status = 500
                return {"error": str(e)}
    
    def run(self, host: str, port: int):
        run(self.app, host=host, port=port, quiet=not DEBUG, server='wsgiref')


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
        if HAS_CPU_MONITOR: self.cpu_monitor = CPUMonitor()
        else: self.cpu_monitor = None
        self.tts_service = TTSService(self.model, Config.DEVICE, self.cpu_monitor)
        self.http_server = HTTPServer(self.tts_service, self)
    
    def warmup(self):
        logger.info("Warming up model...")
        try:
            texts = ["<speak>привет!</speak>",
                     "<speak>как дела?</speak>",
                     "<speak>как погода? азаза. мне нравятся ноги твои и глаза.</speak>"]
            speaker = Config.SPEAKERS[0]["name"]

            for text in texts:
                with torch.no_grad():
                    audio = self.model.apply_tts(text=text, speaker=speaker, sample_rate=24000,
                                put_accent=True, put_yo=True, put_stress_homo=True, put_yo_homo=True)

                    if Config.DEVICE.type == 'cuda': torch.cuda.synchronize()
                    else: _ = audio.numpy()
                    del audio
            for i in range(1000):
                num_to_words(i)
            logger.info("OK")
        except Exception as e:
            logger.error("FAIL")
            raise e
    
    def stop(self, signum=None, frame=None):
        if not self.running: return
        logger.info("Stopping application...")
        
        self.running = False
        
        if self.cpu_monitor:
            self.cpu_monitor.stop()
        
        if self.model:
            try: ModelLoader.unload_model(self.model, Config.DEVICE)
            except: pass
            self.model = None
        
        num_to_words.cache_clear()
        logger.info("OK")
        threading.Timer(1, os._exit(0)).start()
    
    def restart(self):
        if not self.running: return
        logger.info("Restarting application...")
        
        self.running = False
        
        if self.cpu_monitor:
            self.cpu_monitor.stop()
        
        if self.model:
            try: ModelLoader.unload_model(self.model, Config.DEVICE)
            except: pass
            self.model = None
        
        num_to_words.cache_clear()
        time.sleep(1)
        python = sys.executable
        os.execl(python, python, *sys.argv)
        
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
            try:
                kernel32 = ctypes.windll.kernel32
                HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)(self._win_handler)
                kernel32.SetConsoleCtrlHandler(HandlerRoutine, True)
                hStdin = kernel32.GetStdHandle(-10)
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(hStdin, ctypes.byref(mode)):
                    mode.value = (mode.value | 0x0080) & ~0x0040
                    kernel32.SetConsoleMode(hStdin, mode)
            except: pass
        
        self.warmup()
        
        lines = [
            f"Silero TTS Real-Time Server v{MAIN_VERSION} by pav13",
            " GitHub: github.com/pav1388/Silero-TTS-Real-Time-Server",
            f"  Server URL: http://{Config.HOST}:{Config.PORT}"
        ]
        max_len = max(len(line) for line in lines)
        width = max_len + 4
        
        print("")
        print('#' * (width + 2))
        for line in lines:
            print(f"#  {line:<{max_len}}  #")
        print('#' * (width + 2))
        print("")
        print("READY")
        self.running = True
        
        try: self.http_server.run(Config.HOST, Config.PORT)
        except Exception as e: logger.error(f"Server error: {e}")
        finally: self.stop()

if __name__ == "__main__":
    print("")
    print("   ______  _____  _        ______  ______   ______       _______ _______  ______")
    print("  / |       | |  | |      | |     | |  | \\ / |  | \\        | |     | |   / |")
    print("  '------.  | |  | |   _  | |---- | |__| | | |  | |        | |     | |   '------.")
    print("   ____|_/ _|_|_ |_|__|_| |_|____ |_|  \\_\\ \\_|__|_/        |_|     |_|    ____|_/")
    print("")
    Application().run()