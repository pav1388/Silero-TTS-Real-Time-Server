# silero-tts-rt-server.py
# pav13

import ctypes, logging, os, platform, signal, struct, sys, threading, time
import numpy as np
import torch
from bottle import Bottle, hook, request, response, run

# ignore warning for torch 2.0.1
import warnings
warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype")

MAIN_VERSION = "0.8.2"
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
    except ImportError:
        CPUMonitor = None
        logger.warning("'cpu_monitor.py' not found. Speech quality always 'MAX'.")
else:
    CPUMonitor = None
    logger.debug("CPU Monitor disabled. Speech quality always 'MAX'.")

try:
    from text_processor import TextProcessor
except ImportError:
    TextProcessor = None
    logger.warning("'text_processor.py' not found. Num to words, SSML processing and sentence splitting disabled.")


class Config:
    """конфигурация"""
    if CUDA:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device('cpu')
    MODEL_PATH = "models/v5_5_ru.pt"
    MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
    HOST, PORT = "127.0.0.1", 23457
    MAX_SAMPLE_RATE = 48000  # 48000, 24000, 8000
    MAX_TEXT_LENGTH = 900
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


class AudioSynthesizer:
    """генерация звука (синтез речи из SSML)"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
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
            logger.debug(f"  Duration: {duration:.2f}s | Time: {inference_time*1000:.0f}ms | RTF: {inference_time/duration:.3f}")
            
        del audio
        return wav_bytes, duration, inference_time


class TTSService:
    """координация"""
    def __init__(self, model, device, cpu_monitor, text_processor):
        self.text_processor = text_processor
        self.audio_synthesizer = AudioSynthesizer(model, device)
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
        ssml = None
        vol_boost_mod = None
        if self.text_processor is None:
            if len(sentence) > Config.MAX_TEXT_LENGTH:
                ssml = sentence[:Config.MAX_TEXT_LENGTH]
                logger.warning(f"Text length truncated to {Config.MAX_TEXT_LENGTH} chars.")
            else:
                ssml = sentence
            vol_boost_mod = vol_boost
        else:
            ssml, vol_boost_mod = self.text_processor.process_sentence(sentence, speed, pitch, vol_boost)
        
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
        _, speaker = self._resolve_speaker(speaker_id, text)
        speaker_name = speaker['name']
        if self.text_processor is None:
            sentences = [text]
        else:
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
                        logger.debug(f"> Processing sentence {i+1}/{len(sentences)}")
                    
                    quality_config = self._get_quality_config()
                    wav_bytes, duration, inference_time = self._synthesize_sentence(
                        sentence, speaker_name, speed, pitch, vol_boost, quality_config
                    )
                    if DEBUG:
                        total_duration += duration
                        total_time += inference_time
                    
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
        _, speaker = self._resolve_speaker(speaker_id, text)
        speaker_name = speaker['name']
        if self.text_processor is None:
            sentences = [text]
        else:
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
                if DEBUG:
                    total_duration += duration
                    total_time += inference_time
                
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
    
    def synthesize_raw(self, r_count, ssml, speaker="aidar", sample_rate=48000,
                    put_accent=True, put_yo=True, put_stress_homo=True, put_yo_homo=True):
        logger.info(f"Raw request #{r_count}.")
        if DEBUG:
            logger.debug(f"  S:{speaker} S_R:{sample_rate} P_A:{put_accent} P_Y:{put_yo} P_S_H:{put_stress_homo} P_Y_H:{put_yo_homo}")
            logger.debug(f"  SSML_L:{len(ssml)} SSML:{ssml}")
        
        wav_bytes, _, _ = self.audio_synthesizer.synthesize(
            ssml, speaker, sample_rate, put_accent, put_yo, put_stress_homo, put_yo_homo, 0
        )
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
        @self.app.route('/speak/raw', method='OPTIONS')
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
        
        @self.app.route('/speak/raw', method='GET')
        def speak_raw():
            ssml = request.query.text or ""
            speaker = request.query.speaker or "aidar"
            sample_rate = int(request.query.sample_rate or 48000)
            
            if not ssml:
                response.status = 400
                return {"error": "Text is required"}
            self.r_count += 1
              
            def get_bool_param(param_name):
                value = request.query.get(param_name)
                if value is None:
                    return True
                return value.lower() != 'false'
            
            put_accent = get_bool_param('put_accent')
            put_yo = get_bool_param('put_yo')
            put_stress_homo = get_bool_param('put_stress_homo')
            put_yo_homo = get_bool_param('put_yo_homo')
            
            try:
                audio_data = self.tts_service.synthesize_raw(
                    r_count=self.r_count,
                    ssml=ssml,
                    speaker=speaker,
                    sample_rate=sample_rate,
                    put_accent=put_accent,
                    put_yo=put_yo,
                    put_stress_homo=put_stress_homo,
                    put_yo_homo=put_yo_homo
                )
                response.content_type = 'audio/wav'
                response.headers['Content-Length'] = str(len(audio_data))
                return audio_data
                    
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
        self.text_processor = None
        self.running = False
    
    def initialize(self):
        ModelLoader.download_model(Config.MODEL_PATH)
        ModelLoader.setup_torch(Config.DEVICE)
        
        self.model = ModelLoader.load_model(Config.MODEL_PATH, Config.DEVICE)
        self.cpu_monitor = CPUMonitor() if CPUMonitor else None
        self.text_processor = TextProcessor() if TextProcessor else None
        self.tts_service = TTSService(self.model, Config.DEVICE, self.cpu_monitor, self.text_processor)
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
                    audio = self.model.apply_tts(ssml_text=text, speaker=speaker, sample_rate=48000,
                                put_accent=True, put_yo=True, put_stress_homo=True, put_yo_homo=True)

                    if Config.DEVICE.type == 'cuda': torch.cuda.synchronize()
                    else: _ = audio.numpy()
                    del audio
            logger.info("OK")
        except Exception as e:
            logger.error("FAIL")
            raise e
    
    def stop(self, signum=None, frame=None):
        if not self.running: return
        logger.info("Stopping application...")
        
        self.running = False
        
        if self.cpu_monitor: self.cpu_monitor.stop()
        
        if self.model:
            try: ModelLoader.unload_model(self.model, Config.DEVICE)
            except: pass
            self.model = None
        
        logger.info("OK")
        threading.Timer(1, os._exit(0)).start()
    
    def restart(self):
        if not self.running: return
        logger.info("Restarting application...")
        
        self.running = False
        
        if self.cpu_monitor: self.cpu_monitor.stop()
        
        if self.model:
            try: ModelLoader.unload_model(self.model, Config.DEVICE)
            except: pass
            self.model = None
        
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