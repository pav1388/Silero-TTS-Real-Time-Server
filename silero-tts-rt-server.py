# silero-tts-rt-server.py
# pav13

import ctypes, logging, os, platform, re, signal, struct, sys, threading, time
import numpy as np
import psutil
import torch
from bottle import Bottle, hook, request, response, run
from functools import lru_cache
from num2words import num2words
from urllib.parse import unquote

MAIN_VERSION = "0.7.0"
DEBUG = ('--debug' in sys.argv) or (os.environ.get('DEBUG', '0').lower() in ('1', 'true'))
CUDA = ('--cuda' in sys.argv or '--gpu' in sys.argv) or (os.environ.get('CUDA', '0').lower() in ('1', 'true'))

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from cpu_monitor import CPUMonitor
    HAS_CPU_MONITOR = True
except ImportError:
    HAS_CPU_MONITOR = False
    CPUMonitor = None
    logger.warning("cpu_monitor.py not found. Speech quality always 'MAX'.")


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
            "put_stress_homo": True,  "put_yo_homo": True,  "name": "CPU Monitor disabled"}
    SPEAKERS = []
    REAL_SPEAKERS_COUNT = 0
    SPEAKERS_INFO = {
        "aidar": {"gender": "male", "lang": "ru"},
        "baya": {"gender": "female", "lang": "ru"},
        "kseniya": {"gender": "female", "lang": "ru"},
        "eugene": {"gender": "male", "lang": "ru"},
        "xenia": {"gender": "female", "lang": "ru"},
    }
    

class ModelLoader:
    """загрузка модели и инициализация torch"""
    @staticmethod
    def setup_torch(device):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        if device.type == 'cpu':
            try:
                cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
            except:
                cores = os.cpu_count() or 1
            n_threads = 2 if cores > 1 else 1
            torch.set_num_threads(n_threads)
            torch.set_num_interop_threads(1)
            logger.info(f"Device: CPU | Threads: {n_threads} | Cores: {cores}")
        
        elif device.type == 'cuda':
            logger.info(f"Device: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            
            if DEBUG:
                logger.info(f"CUDA: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")
                props = torch.cuda.get_device_properties(0)
                logger.info(f"Capability: {props.major}.{props.minor} | SMs: {props.multi_processor_count}")
        
        if DEBUG:
            try:
                mem = psutil.virtual_memory()
                logger.info(f"RAM: {mem.total/1e9:.1f} GB | Free: {mem.available/1e9:.1f} GB")
                logger.info(f"PyTorch: {torch.__version__} | Python: {sys.version.split()[0]}")
                logger.info(f"OS: {platform.system()} {platform.release()} | Arch: {platform.machine()}")
            except:
                pass
    
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
                 "lang": Config.SPEAKERS_INFO.get(name, {}).get("lang", "n/d")}
                for idx, name in enumerate(model_speakers)
            ]
            
            Config.REAL_SPEAKERS_COUNT = len(model_speakers)
            
            special_speakers = [
                {"id": len(model_speakers),     "name": "RANDOM",    "gender": "both",   "lang": "ru"},
                {"id": len(model_speakers) + 1, "name": "RANDOM_M",  "gender": "male",   "lang": "ru"},
                {"id": len(model_speakers) + 2, "name": "RANDOM_F",  "gender": "female", "lang": "ru"},
                {"id": len(model_speakers) + 3, "name": "HASH",      "gender": "both",   "lang": "ru"},
            ]
            Config.SPEAKERS.extend(special_speakers)
            
            if DEBUG:
                logger.debug(f"Loaded {len(model_speakers)} speakers from model:")
                for spk in Config.SPEAKERS[:len(model_speakers)]:
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
    """обработка текста (SSML, числа, транслитерация)"""
    pause0, pause1, pause2, pause3, pause4, pause5 = 0, 130, 180, 215, 320, 480
    BREAK_TIME_MAP = {'.': pause4, ',': pause2, '!': pause4, '?': pause4, 
                      '(': pause2, ')': pause2, '[': pause2, ']': pause2, 
                      ':': pause1, ';': pause3, '—': pause3, '…': pause5}
    EMOTIONS = {'!': (7, 0), '?': (-7, 0)} # 'знак': (speed в %, pitch от -2 до 2)
    ALLOWED = frozenset("_~абвгдеёжзийклмнопрстуфхцчшщъыьэюя +.,!?…:;–")
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

    def process_text(self, text: str) -> str:
        if not text: return "", 0
        len_text = len(text)
        if len_text > Config.MAX_TEXT_LENGTH:
            len_text = Config.MAX_TEXT_LENGTH
            text = text[:len_text]
            logger.info(f"Text length truncated to {len_text} chars.")
        
        text = unquote(text).lower()
        has_latin = any(ch in self.LATIN for ch in text)
        processed_body = self._proc(text, len_text, has_latin)
        return f'<speak><prosody rate="{self.speed_percent}%" pitch="{self.pitch_level}">{processed_body}</prosody></speak>', len_text

    def _proc(self, text: str, len_text: int, has_latin: bool) -> str:
        res, buf, i, n = [], [], 0, len_text
        BREAK_TIME_MAP, ALLOWED, LATIN, EMOTIONS = self.BREAK_TIME_MAP, self.ALLOWED, self.LATIN, self.EMOTIONS
        while i < n:
            ch = text[i]
            if ch.isdigit():
                i, p = self._num(text, i)
                buf.append(p)
                continue
            if has_latin and ch in LATIN:
                ni, tr = self._trans(text, i)
                if tr and tr != ch: buf.append(tr)
                i = ni
                # word_start = i
                # while i < n and text[i] in LATIN: i += 1
                # lat = text[word_start:i]
                # cyr = translit(lat, 'ru')
                # if cyr: buf.append(cyr)
                continue
            if ch in BREAK_TIME_MAP:
                skip = 1
                if ch == '.' and i + 2 < n and text[i+1] == '.':
                    if text[i+2] == '.':
                        ch = '…'
                        skip = 3
                if buf:
                    s = ''.join(buf).strip()
                    if ch in EMOTIONS: text_to_wrap = s + ' ' + ch
                    else: text_to_wrap = s
                    res.append(self._wrap(text_to_wrap, ch))
                res.append(f'<break time="{BREAK_TIME_MAP[ch]}ms"/>')
                buf.clear()
                i += skip
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
            if s: res.append(self._wrap(s, None))
        return ''.join(res).strip()
    
    def _num(self, text: str, start: int) -> tuple:
        i, n = start, len(text)
        while i < n and text[i].isdigit(): i += 1
        
        if i == start:
            return start + 1, text[start]
        
        num = int(text[start:i])
        num_word = num_to_words(num)
        
        if i < n and text[i] == '%':
            if num % 10 == 1 and num % 100 != 11:
                percent_form = 'процент'
            elif 2 <= num % 10 <= 4 and (num % 100 < 10 or num % 100 >= 20):
                percent_form = 'процента'
            else:
                percent_form = 'процентов'
            
            return i + 1, f"{num_word} {percent_form}"
        
        if i < n and text[i] in '.,' and i + 1 < n and text[i+1].isdigit():
            k = i + 1
            while k < n and text[k].isdigit(): k += 1
            fractional = text[i+1:k]
            fractional_word = num_to_words(int(fractional))
            
            if num % 10 == 1 and num % 100 != 11:
                whole_word = 'целая'
            else:
                whole_word = 'целых'
            
            fractional_len = len(fractional)
            if fractional_len == 1:
                fractional_part = 'десятых'
            elif fractional_len == 2:
                fractional_part = 'сотых'
            elif fractional_len == 3:
                fractional_part = 'тысячных'
            elif fractional_len == 4:
                fractional_part = 'десятитысячных'
            elif fractional_len == 5:
                fractional_part = 'стотысячных'
            else:
                fractional_part = ''
            
            return k, f"{num_word} {whole_word} {fractional_word} {fractional_part}"
        
        if i < n and text[i] == '/' and i + 1 < n and text[i+1].isdigit():
            k = i + 1
            while k < n and text[k].isdigit(): k += 1
            denominator = int(text[i+1:k])
            denominator_word = num_to_words(denominator)
            
            return k, f"{num_word} дробь {denominator_word}"
        
        return i, num_word

    def _trans(self, text: str, pos: int) -> tuple:
        node, best, best_pos = self.transl_trie, None, pos
        j = pos
        n = len(text)
        while j < n and text[j] in node:
            node = node[text[j]]; j += 1
            if '_' in node: best, best_pos = node['_'], j
        return (best_pos, best) if best else (pos + 1, text[pos] if text[pos] in self.ALLOWED else " ")
    
    def _wrap(self, text: str, end_punct: str) -> str:
        if not text: return ""
        if end_punct not in self.EMOTIONS: return text
        
        sm, pd = self.EMOTIONS[end_punct]
        emo_r = f"{int(self.speed_percent + sm)}"
        pitch_order = Config.PITCH_ORDER
        try:
            idx = max(0, min(len(pitch_order) - 1, pitch_order.index(self.pitch_level) + pd))
            emo_p = pitch_order[idx]
        except ValueError:
            emo_p = self.pitch_level
        
        def attrs(rate, pitch): return f'rate="{rate}%" pitch="{pitch}"'
        
        if ' ' not in text:
            return f'<prosody {attrs(emo_r, emo_p)}>{text}</prosody>'
        
        words = text.split()
        
        if len(words) < 4:
            return f'<prosody {attrs(emo_r, emo_p)}>{text}</prosody>'
        
        tail_count = max(1, int(len(words) * 0.2))
        
        if tail_count < len(words) and words[-1] in ['!', '?'] and tail_count == 1:
             tail_count = 2

        head_words = words[:-tail_count]
        tail_words = words[-tail_count:]
        head_text = " ".join(head_words)
        tail_text = " ".join(tail_words)
        return f'{head_text} <prosody {attrs(emo_r, emo_p)}>{tail_text}</prosody>'


@lru_cache(maxsize=2048)
def num_to_words(num: int) -> str:
    """преобразование чисел в слова"""
    return num2words(num, lang='ru')


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
                put_accent: bool, put_yo: bool, put_stress_homo: bool, put_yo_homo: bool, vol_boost: float) -> bytes:
        try:
            with torch.no_grad():
                if DEBUG and self.device.type == 'cuda':
                    torch.cuda.synchronize()
            
                audio = self.model.apply_tts(
                    ssml_text=ssml, speaker=speaker_name, sample_rate=sample_rate, 
                    put_accent=put_accent, put_yo=put_yo,
                    put_stress_homo=put_stress_homo, put_yo_homo=put_yo_homo)
            
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
    
    def synthesize_stream(self, text, speaker_id, speed, pitch, vol_boost, r_count):
        resolved_speaker_id, speaker = self._resolve_speaker(speaker_id, text)
        sentences = re.split(r'(?<=[.!?…])\s+', text)
        result = []
        append = result.append

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue

            if len(sentence) <= 200:
                append(sentence)
                continue

            parts = re.split(r'(?<=[,;:—])\s+', sentence)
            current = []
            cur_len = 0
            cur_join = " ".join

            for part in parts:
                part = part.strip()
                if not part: continue

                if cur_len and cur_len < 60:
                    current.append(part)
                    cur_len += len(part) + 1
                elif current:
                    append(cur_join(current))
                    current = [part]
                    cur_len = len(part)
                else:
                    current = [part]
                    cur_len = len(part)

            if current:
                if result and cur_len < 60:
                    result[-1] += " " + cur_join(current)
                else:
                    append(cur_join(current))

        sentences = result

        if not sentences:
            raise ValueError("No valid sentences found")
        
        logger.info(f"Stream request #{r_count}: {len(sentences)} sentences.")
        
        if DEBUG:
            for i, s in enumerate(sentences):
                logger.debug(f"  [{i+1}] {s[:80]}{'...' if len(s) > 80 else ''}")

        def generate():
            first = True
            for i, sentence in enumerate(sentences):
                try:
                    if DEBUG:
                        logger.debug(f"Synth {i+1}/{len(sentences)}")

                    wav = self._synthesize_speech(
                        sentence, resolved_speaker_id, speaker, speed, pitch, vol_boost
                    )

                    if len(wav) > 44 and wav[:4] == b'RIFF':
                        offset = 12
                        while offset < len(wav) - 8:
                            chunk_id = wav[offset:offset+4]
                            chunk_size = struct.unpack('<I', wav[offset+4:offset+8])[0]

                            if chunk_id == b'data':
                                if first:
                                    header = bytearray(wav[:offset+8])
                                    struct.pack_into('<I', header, 4, 0xFFFFFFFF)
                                    struct.pack_into('<I', header, offset+4, 0xFFFFFFFF)
                                    yield bytes(header)
                                    first = False

                                yield wav[offset+8:offset+8+chunk_size]
                                break

                            offset += 8 + chunk_size
                    else:
                        yield wav

                except Exception as e:
                    logger.error(f"Error sentence {i+1}: {e}")
                    continue
            
            logger.debug("All sentences processed")
        
        return generate()
    
    def synthesize_once(self, text, speaker_id, speed, pitch, vol_boost, r_count):
        resolved_speaker_id, speaker = self._resolve_speaker(speaker_id, text)
        
        logger.info(f"Speech request #{r_count}.")
        
        audio_data = self._synthesize_speech(
            text, resolved_speaker_id, speaker, speed, pitch, vol_boost
        )
        return audio_data

    def _synthesize_speech(self, text: str, speaker_id: int, speaker: dict, 
                                 speed_percent: int, pitch_level: str, vol_boost: float) -> bytes:
        t_start = time.time() if DEBUG else None
        
        if self.cpu_monitor:
            self.cpu_monitor.record_activity()
            q = self.cpu_monitor.get_current_quality_config()
        else:
            q = Config.MAX_QUALITY_CONFIG
        
        sr = min(Config.MAX_SAMPLE_RATE, q["sample_rate"])
        
        self.text_processor.set_ssml_params(speed_percent, pitch_level)
        ssml, len_text = self.text_processor.process_text(text)
        
        if DEBUG: 
            logger.debug(f"N:{speaker['name']} S:{speed_percent}% P:{pitch_level} Vol:{vol_boost}dB SR:{sr} Q:{q['name']}")
            logger.debug(f"L:{len_text} SSML:{ssml}")
        
        wav_bytes = self.audio_synthesizer.synthesize(
            ssml, speaker['name'], sr,
            q['put_accent'], q['put_yo'], q['put_stress_homo'], q['put_yo_homo'],
            vol_boost
        )
        
        if DEBUG and t_start:
            num_samples = len(wav_bytes) - 44
            dur = num_samples / (sr * 2)
            if dur > 0: 
                logger.debug(f"T:{(time.time()-t_start)*1000:.0f}ms D:{dur:.2f}s RTF:{(time.time()-t_start)/dur:.2f}")
        
        return wav_bytes


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
            CUDA = Config.DEVICE.type == 'cuda'
            tp = TextProcessor()
            tp.set_ssml_params(100, "medium")
            texts = ["привет", "как дела?", "это прогрев модели для устранения задержки"]
            speaker = Config.SPEAKERS[0]["name"]

            for text in texts:
                ssml, _ = tp.process_text(text)

                with torch.no_grad():
                    audio = self.model.apply_tts(ssml_text=ssml, speaker=speaker, sample_rate=48000,
                                put_accent=True, put_yo=True, put_stress_homo=True, put_yo_homo=True)

                    if CUDA: torch.cuda.synchronize()
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
        print('=' * 60)
        print(f"  Silero TTS Real-Time Server v{MAIN_VERSION} by pav13")
        print("   GitHub: github.com/pav1388/Silero-TTS-Real-Time-Server")
        print(f"    Server URL: http://{Config.HOST}:{Config.PORT}")
        if DEBUG: print("  DEBUG mode ON")
        print('=' * 60)
        print("READY")
        self.running = True
        
        try: self.http_server.run(Config.HOST, Config.PORT)
        except Exception as e: logger.error(f"Server error: {e}")
        finally: self.stop()

if __name__ == "__main__":
    print()
    print("   ______  _____  _        ______  ______   ______       _______ _______  ______")
    print("  / |       | |  | |      | |     | |  | \\ / |  | \\        | |     | |   / |")
    print("  '------.  | |  | |   _  | |---- | |__| | | |  | |        | |     | |   '------.")
    print("   ____|_/ _|_|_ |_|__|_| |_|____ |_|  \\_\\ \\_|__|_/        |_|     |_|    ____|_/")
    print()
    Application().run()