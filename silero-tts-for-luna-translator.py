# silero-tts-for-luna-translator.py

import io
import os
import sys
import time
import struct
import torch
import traceback
import numpy as np
import psutil
from bottle import Bottle, request, response, run
from num2words import num2words
from urllib.parse import unquote
from functools import lru_cache
from queue import Queue, Empty
from threading import Lock, Thread
from collections import deque
import threading
import uuid
import math

# ==================== КОНФИГУРАЦИЯ ====================

class Config:
    MODEL_PATH = "models/v5_5_ru.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAMPLE_RATE = 48000
    HOST = "127.0.0.1"
    PORT = 5000
    REQUEST_TIMEOUT = 15  # ожидание ответа от модели, секунд
    MAX_QUEUE_SIZE = 5  # Максимальный размер очереди
    MAX_TEXT_LENGTH = 950  # Максимальная длина текста
    
    # Настройки динамической скорости
    SPEED_ADJUSTMENT_INTERVAL = 1.0  # Интервал обновления скорости, секунд
    SPEED_SMOOTHING_FACTOR = 0.4  # Коэффициент сглаживания (0-1, чем выше, тем быстрее реакция)
    BASE_SPEED = 1.0  # Базовая скорость (когда очередь пуста)
    MAX_QUEUE_FACTOR = 3.0  # Максимальное ускорение при полной очереди
    SPEED_DECAY_RATE = 3.0  # Скорость уменьшения скорости (быстрее чем увеличение)
    
    # Настройки снижения качества при нагрузке CPU
    CPU_MONITOR_INTERVAL = 0.5  # Интервал измерения CPU (секунды)
    CPU_HIGH_THRESHOLD = 80.0  # Порог высокой нагрузки (%)
    CPU_CRITICAL_THRESHOLD = 95.0  # Порог критической нагрузки (%)
    CPU_SAMPLE_DURATION = 0.1  # Длительность замера CPU (сек)
    
    # Доступные конфигурации качества генерации
    QUALITY_LEVELS = [
        {"sample_rate": 8000, "put_accent": False, "put_yo": False, "name": "LOWEST"},      # Уровень 0
        {"sample_rate": 8000, "put_accent": True, "put_yo": False, "name": "LOW"},        # Уровень 1
        {"sample_rate": 24000, "put_accent": False, "put_yo": False, "name": "MEDIUM"},      # Уровень 2
        {"sample_rate": 24000, "put_accent": True, "put_yo": True, "name": "HIGH"},       # Уровень 3
        {"sample_rate": 48000, "put_accent": True, "put_yo": False, "name": "VERY_HIGH"},   # Уровень 4
        {"sample_rate": 48000, "put_accent": True, "put_yo": True, "name": "MAXIMUM"}       # Уровень 5
    ]
    
class AudioPauses:
    # паузы в мс после
    SENTENCE = 320  # точки
    COMMA = 220  # запятой
    SENTENCE_END = 400  # в конце реплики

# ==================== ДАННЫЕ О ГОЛОСАХ ====================

SPEAKERS = [
    {"id": 0, "name": "aidar", "style": "male", "lang": ["ru"]},
    {"id": 1, "name": "baya", "style": "female", "lang": ["ru"]},
    {"id": 2, "name": "kseniya", "style": "female", "lang": ["ru"]},
    {"id": 3, "name": "xenia", "style": "female", "lang": ["ru"]},
    {"id": 4, "name": "eugene", "style": "male", "lang": ["ru"]},
]

# Расширенный словарь с параметрами голосов
VOICE_SETTINGS = {
    "aidar": {"volume_boost": 3, "pitch": "high", "base_speed": 1.1},
    "eugene": {"volume_boost": 0.5, "pitch": "low", "base_speed": 0.9},
    "baya": {"volume_boost": 0, "pitch": "low", "base_speed": 1.0},
    "kseniya": {"volume_boost": 0, "pitch": "low", "base_speed": 1.0},
    "xenia": {"volume_boost": 1, "pitch": "medium", "base_speed": 0.95}
}

# ==================== МОНИТОР CPU ====================

class CPUMonitor:
    """Щадящий мониторинг нагрузки CPU в отдельном потоке"""
    
    def __init__(self):
        self.current_load = 0.0
        self.current_quality_level = len(Config.QUALITY_LEVELS) - 1  # Начинаем с максимального
        self.quality_stable_since = time.time()
        self.last_change_time = 0
        self.max_level = len(Config.QUALITY_LEVELS) - 1

        self.lock = Lock()
        self.running = True
        
        # История нагрузки для сглаживания
        self.load_history = deque(maxlen=3)
        
        # Минимальный интервал между изменениями качества (2 секунды)
        self.min_change_interval = 2.0
        
        self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _get_cpu_load_gentle(self) -> float:
        """Щадящее измерение CPU нагрузки"""
        try:
            cpu_percent = psutil.cpu_percent(interval=Config.CPU_SAMPLE_DURATION)
            return cpu_percent
        except Exception as e:
            print(f"[CPU Monitor] Error: {e}")
            return 0.0
    
    def _get_average_load(self) -> float:
        """Получить среднюю нагрузку за последние измерения"""
        if not self.load_history:
            return self.current_load
        return sum(self.load_history) / len(self.load_history)
    
    def _calculate_target_quality(self, avg_load: float) -> int:
        """Расчет целевого уровня качества на основе нагрузки"""
        
        # Плавное определение целевого уровня в зависимости от нагрузки
        if avg_load >= Config.CPU_CRITICAL_THRESHOLD:
            return 0
        elif avg_load >= Config.CPU_HIGH_THRESHOLD:
            load_ratio = (avg_load - Config.CPU_HIGH_THRESHOLD) / (Config.CPU_CRITICAL_THRESHOLD - Config.CPU_HIGH_THRESHOLD)
            reduction = int(load_ratio * 5)  # Максимум снижение на 5 уровней
            target = max(0, self.max_level - reduction)
            return target
        else:
            return self.max_level
    
    def _monitor_loop(self):
        """Цикл мониторинга CPU"""
        while self.running:
            try:
                # Измеряем нагрузку
                cpu_load = self._get_cpu_load_gentle()
                
                with self.lock:
                    # Добавляем в историю
                    self.load_history.append(cpu_load)
                    avg_load = self._get_average_load()
                    self.current_load = avg_load
                    
                    # Рассчитываем целевой уровень качества
                    target_level = self._calculate_target_quality(avg_load)
                    
                    now = time.time()
                    
                    # Изменяем качество только если:
                    # 1. Целевой уровень отличается от текущего
                    # 2. Прошло достаточно времени с последнего изменения
                    if target_level != self.current_quality_level and \
                       now - self.last_change_time >= self.min_change_interval:
                        
                        old_level = self.current_quality_level
                        old_config = Config.QUALITY_LEVELS[old_level]
                        
                        # Постепенное изменение (на 1 уровень за раз)
                        if target_level > self.current_quality_level:
                            self.current_quality_level = min(self.current_quality_level + 1, self.max_level)
                        else:
                            self.current_quality_level = max(self.current_quality_level - 1, 0)
                        
                        new_config = Config.QUALITY_LEVELS[self.current_quality_level]
                        self.last_change_time = now
                        self.quality_stable_since = now
                        
                        # Логируем изменение
                        direction = "↑" if self.current_quality_level > old_level else "↓"
                        print(f"[CPU] {direction} LOAD {avg_load:.1f}% → {old_config['name']} → {new_config['name']}")
                        print(f"      {new_config['sample_rate']}Hz, Accent: {new_config['put_accent']}, Yo: {new_config['put_yo']}")
                
                time.sleep(Config.CPU_MONITOR_INTERVAL)
                
            except Exception as e:
                print(f"[CPU Monitor Error] {e}")
                time.sleep(1)
    
    def get_current_quality_config(self) -> dict:
        """Получить текущую конфигурацию качества"""
        with self.lock:
            return Config.QUALITY_LEVELS[self.current_quality_level].copy()
    
    def get_current_sample_rate(self) -> int:
        """Получить текущую частоту дискретизации"""
        with self.lock:
            return Config.QUALITY_LEVELS[self.current_quality_level]["sample_rate"]
    
    def get_put_accent(self) -> bool:
        """Получить текущее значение put_accent"""
        with self.lock:
            return Config.QUALITY_LEVELS[self.current_quality_level]["put_accent"]
    
    def get_put_yo(self) -> bool:
        """Получить текущее значение put_yo"""
        with self.lock:
            return Config.QUALITY_LEVELS[self.current_quality_level]["put_yo"]
    
    def get_cpu_load(self) -> float:
        """Получить текущую нагрузку CPU"""
        with self.lock:
            return self.current_load
    
    def stop(self):
        """Остановка монитора"""
        self.running = False

# ==================== УТИЛИТЫ ====================

@lru_cache(maxsize=1024)
def num_to_words(num: str) -> str:
    """Конвертация числа в слова"""
    if not num or not num.isdigit():
        return str(num) if num else ""
    if len(num) > 9:
        return num
    return num2words(int(num), lang='ru')

class TextProcessor:
    """Обработка текста: числа, транслитерация, пунктуация"""
    
    def __init__(self):
        self._init_trie()
        self._init_punctuation_config()
        self.allowed_chars = frozenset("_~абвгдеёжзийклмнопрстуфхцчшщъыьэюя +.,!?…:;–")
        self.latin_letters = frozenset("abcdefghijklmnopqrstuvwxyz")
        
    def _init_trie(self):
        TRANSLIT_MAP = {
            'ough': 'о', 'augh': 'о', 'eigh': 'эй', 'tion': 'шн', 'shch': 'щ',
            'tch': 'ч', 'sch': 'ск', 'scr': 'скр', 'thr': 'зр', 'squ': 'скв',
            'ear': 'ир', 'air': 'эр', 'are': 'эр', 'the': 'зэ', 'and': 'энд',
            'ea': 'и', 'ee': 'и', 'oo': 'у', 'ai': 'эй', 'ay': 'эй', 'ei': 'эй',
            'ey': 'эй', 'oi': 'ой', 'oy': 'ой', 'ou': 'ау', 'ow': 'ау', 'au': 'о',
            'aw': 'о', 'ie': 'и', 'ui': 'у', 'ue': 'ю', 'uo': 'уо', 'eu': 'ю',
            'ew': 'ю', 'oa': 'о', 'oe': 'о', 'sh': 'ш', 'ch': 'ч', 'zh': 'ж',
            'th': 'з', 'kh': 'х', 'ti': 'тай', 'ts': 'ц', 'ph': 'ф', 'wh': 'в',
            'gh': 'г', 'qu': 'кв', 'gu': 'г', 'dg': 'дж', 'ce': 'це', 'ci': 'си',
            'cy': 'си', 'ck': 'к', 'ge': 'дж', 'gi': 'джи', 'gy': 'джи', 'er': 'эр',
            'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е', 'f': 'ф', 'g': 'г',
            'h': 'х', 'i': 'и', 'j': 'дж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
            'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т', 'u': 'у',
            'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'й', 'z': 'з',
        }
        
        self.translit_trie = {}
        for key, value in TRANSLIT_MAP.items():
            node = self.translit_trie
            for ch in key:
                if ch not in node:
                    node[ch] = {}
                node = node[ch]
            node['_'] = value
    
    def _init_punctuation_config(self):
        self.punctuation_config = {
            '.': {'pause': AudioPauses.SENTENCE, 'end_sentence': True},
            '!': {'pause': AudioPauses.SENTENCE, 'end_sentence': True},
            '?': {'pause': AudioPauses.SENTENCE, 'end_sentence': True},
            '(': {'pause': AudioPauses.COMMA, 'end_sentence': False},
            ')': {'pause': AudioPauses.COMMA, 'end_sentence': False},
            ',': {'pause': AudioPauses.COMMA, 'end_sentence': False},
            ';': {'pause': AudioPauses.COMMA, 'end_sentence': False},
            ':': {'pause': AudioPauses.COMMA // 2, 'end_sentence': False},
        }
    
    def process_text(self, text: str, add_final_pause: bool = True) -> str:
        if not text:
            return ""
        
        text = unquote(text).lower()
        
        result_parts = []
        i = 0
        n = len(text)
        has_latin = any(ch in self.latin_letters for ch in text)
        
        ends_with_sentence = False
        last_was_space = False
        
        while i < n:
            ch = text[i]
            
            # Пропускаем спецсимволы
            # if ch in '#+%':
                # i += 1
                # continue
            
            # 1. ЧИСЛА
            if ch.isdigit():
                i, number_result = self._process_number(text, i)
                result_parts.append(number_result)
                ends_with_sentence = False
                last_was_space = False
                continue
            
            # 2. ТРАНСЛИТЕРАЦИЯ
            if has_latin and ch in self.latin_letters:
                new_i, translit_result = self._process_transliteration(text, i)
                if translit_result and translit_result != ch:
                    result_parts.append(translit_result)
                    i = new_i
                    ends_with_sentence = False
                    last_was_space = False
                    continue
            
            # 3. ПУНКТУАЦИЯ
            if ch in self.punctuation_config:
                config = self.punctuation_config[ch]
                if result_parts and result_parts[-1] == ' ':
                    result_parts.pop()
                
                result_parts.append(ch)
                result_parts.append(f'<break time="{config["pause"]}ms"/> ')
                ends_with_sentence = config['end_sentence']
                last_was_space = True
                i += 1
                continue
            
            # 4. ПРОБЕЛЫ
            if ch.isspace() or ch == ' ':
                if not last_was_space:
                    result_parts.append(' ')
                    last_was_space = True
                i += 1
                continue
            
            # 5. РАЗРЕШЕННЫЕ СИМВОЛЫ
            if ch in self.allowed_chars:
                result_parts.append(ch)
                ends_with_sentence = False
                last_was_space = False
                i += 1
                continue
            
            # 6. МУСОР -> ПРОБЕЛ
            if not last_was_space:
                result_parts.append(' ')
                last_was_space = True
            i += 1
        
        result = ''.join(result_parts).strip()
        
        if add_final_pause and ends_with_sentence:
            result += f'<break time="{AudioPauses.SENTENCE_END}ms"/>'
        
        return result
    
    def _process_number(self, text: str, start: int) -> tuple:
        i = start
        n = len(text)
        j = i
        while j < n and text[j].isdigit():
            j += 1
        
        if j == i:
            return i + 1, text[i]
        
        num1 = text[i:j]
        
        # Десятичная дробь
        if j < n and text[j] in '.,' and j + 1 < n and text[j + 1].isdigit():
            k = j + 1
            while k < n and text[k].isdigit():
                k += 1
            num2 = text[j + 1:k]
            return k, f"{num_to_words(num1)} точка {num_to_words(num2)}"
        
        # Обычная дробь
        if j < n and text[j] == '/' and j + 1 < n and text[j + 1].isdigit():
            k = j + 1
            while k < n and text[k].isdigit():
                k += 1
            num2 = text[j + 1:k]
            return k, f"{num_to_words(num1)} дробь {num_to_words(num2)}"
        
        return j, num_to_words(num1)
    
    def _process_transliteration(self, text: str, pos: int) -> tuple:
        node = self.translit_trie
        best_match = None
        best_pos = pos
        j = pos
        
        while j < len(text) and text[j] in node:
            node = node[text[j]]
            j += 1
            if '_' in node:
                best_match = node['_']
                best_pos = j
        
        if best_match:
            return best_pos, best_match
        
        return pos + 1, text[pos] if text[pos] in self.allowed_chars else " "

# ==================== ОБРАБОТЧИК АУДИО ====================

class AudioProcessor:
    def __init__(self, model, device, cpu_monitor):
        self.model = model
        self.device = device
        self.text_processor = TextProcessor()
        self.cpu_monitor = cpu_monitor
    
    def get_voice_settings(self, speaker_name: str) -> dict:
        """Получение настроек голоса из словаря"""
        if speaker_name in VOICE_SETTINGS:
            return VOICE_SETTINGS[speaker_name]
        return {"volume_boost": 0, "pitch": "medium", "base_speed": 1.0}
    
    def map_pitch_to_value(self, pitch_setting: str) -> float:
        """Преобразование настройки pitch в числовое значение"""
        pitch_map = {
            "x-low": -10,
            "low": -4,
            "medium": 0,
            "high": 4,
            "x-high": 10
        }
        return pitch_map.get(pitch_setting, 0)
    
    def map_value_to_pitch(self, pitch_value: float) -> str:
        """Преобразование числового значения pitch в строку для SSML"""
        if pitch_value <= -10:
            return "x-low"
        elif pitch_value <= -4:
            return "low"
        elif pitch_value >= 10:
            return "x-high"
        elif pitch_value >= 4:
            return "high"
        else:
            return "medium"
    
    def synthesize(self, text: str, speaker: str, dynamic_speed: float, 
                   pitch_adjustment: float, volume_adjustment: float, 
                   requested_sample_rate: int) -> bytes:
        try:
            # Получаем актуальную конфигурацию качества от монитора CPU
            quality_config = self.cpu_monitor.get_current_quality_config()
            
            # Частота дискретизации (не может быть выше запрошенной)
            sample_rate = min(requested_sample_rate, quality_config["sample_rate"])
            
            # Получаем настройки для put_accent и put_yo
            put_accent = quality_config["put_accent"]
            put_yo = quality_config["put_yo"]
            
            # Получаем базовые настройки голоса
            voice_settings = self.get_voice_settings(speaker)
            
            # Финальная скорость = базовая скорость голоса * динамическая скорость
            final_speed = voice_settings["base_speed"] * dynamic_speed
            final_speed = max(0.1, min(10.0, final_speed))
            rate_str = f"{int(final_speed * 100)}%"
            
            # Pitch: базовая настройка голоса + внешняя регулировка
            base_pitch_value = self.map_pitch_to_value(voice_settings["pitch"])
            final_pitch_value = base_pitch_value + pitch_adjustment
            final_pitch_value = max(-10, min(10, final_pitch_value))
            pitch_str = self.map_value_to_pitch(final_pitch_value)
            
            # Volume: базовая громкость голоса + внешняя регулировка
            final_volume = voice_settings["volume_boost"] + volume_adjustment
            
            # Обработка текста
            processed_text = self.text_processor.process_text(text)
            if not processed_text:
                processed_text = text
            
            # SSML
            ssml = f'<speak><prosody rate="{rate_str}" pitch="{pitch_str}">{processed_text}</prosody></speak>'
            
            cpu_load = self.cpu_monitor.get_cpu_load()
            quality_name = quality_config["name"]
            
            print(f"[DEBUG] Quality: {quality_name} | CPU: {cpu_load:.1f}% | Sample rate: {sample_rate}Hz")
            print(f"[DEBUG] put_accent={put_accent}, put_yo={put_yo} | Speaker: {speaker}")
            print(f"[DEBUG] Final speed: {final_speed:.2f}x (base: {voice_settings['base_speed']}, dynamic: {dynamic_speed:.2f})")
            
            # Генерация аудио с динамическими параметрами качества
            audio = self.model.apply_tts(
                ssml_text=ssml,
                speaker=speaker,
                sample_rate=sample_rate,
                put_accent=put_accent,
                put_yo=put_yo
            )
            
            audio_np = audio.cpu().numpy() if hasattr(audio, 'cpu') else np.array(audio, dtype=np.float32)
            
            # Применение громкости
            if final_volume != 0:
                volume_factor = 10 ** (final_volume / 20.0)
                audio_np = np.clip(audio_np * volume_factor, -1.0, 1.0)
            
            # Конвертация в WAV
            audio_int16 = (audio_np * 32767).astype(np.int16)
            return self._numpy_to_wav_bytes(audio_int16, sample_rate)
            
        except Exception as e:
            print(f"[X] Ошибка синтеза: {e}")
            traceback.print_exc()
            # Возврат короткой тишины
            silent_audio = np.zeros(int(16000 * 0.1), dtype=np.float32)
            audio_int16 = (silent_audio * 32767).astype(np.int16)
            return self._numpy_to_wav_bytes(audio_int16, 16000)
    
    def _numpy_to_wav_bytes(self, audio_int16: np.ndarray, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        channels = 1
        bits_per_sample = 16
        bytes_per_sample = bits_per_sample // 8
        byte_rate = sample_rate * channels * bytes_per_sample
        block_align = channels * bytes_per_sample
        data_size = len(audio_int16) * bytes_per_sample
        file_size = 36 + data_size
        
        buffer.write(b'RIFF')
        buffer.write(struct.pack('<I', file_size))
        buffer.write(b'WAVE')
        buffer.write(b'fmt ')
        buffer.write(struct.pack('<I', 16))
        buffer.write(struct.pack('<H', 1))
        buffer.write(struct.pack('<H', channels))
        buffer.write(struct.pack('<I', sample_rate))
        buffer.write(struct.pack('<I', byte_rate))
        buffer.write(struct.pack('<H', block_align))
        buffer.write(struct.pack('<H', bits_per_sample))
        buffer.write(b'data')
        buffer.write(struct.pack('<I', data_size))
        buffer.write(audio_int16.tobytes())
        
        return buffer.getvalue()

# ==================== МЕНЕДЖЕР ОЧЕРЕДЕЙ С ДИНАМИЧЕСКОЙ СКОРОСТЬЮ ====================

class DynamicSpeedController:
    """Контроллер динамического изменения скорости на основе загруженности очереди"""
    
    def __init__(self):
        self.current_speed_factor = Config.BASE_SPEED
        self.target_speed_factor = Config.BASE_SPEED
        self.last_update_time = time.time()
        self.lock = Lock()
        self.queue_sizes = deque(maxlen=10)  # Храним историю размеров очереди для сглаживания
        self.last_reported_speed = Config.BASE_SPEED
        self.report_threshold = 0.15  # Печатать только при изменении скорости > 15%
    
    def update_queue_size(self, queue_size: int, max_queue_size: int):
        """Обновление информации о размере очереди"""
        with self.lock:
            self.queue_sizes.append(queue_size)
            avg_queue_size = sum(self.queue_sizes) / len(self.queue_sizes)
            load_factor = min(1.0, avg_queue_size / max_queue_size)
            speed_multiplier = 1.0 + (load_factor ** 1.5) * (Config.MAX_QUEUE_FACTOR - 1.0)
            self.target_speed_factor = speed_multiplier
            if abs(speed_multiplier - self.last_reported_speed) > self.report_threshold:
                print(f"[SPEED] {self.last_reported_speed:.2f}x → {speed_multiplier:.2f}x (load: {load_factor:.0%}, queue: {queue_size})")
                self.last_reported_speed = speed_multiplier
    
    def get_current_speed(self) -> float:
        """Получение текущей сглаженной скорости"""
        with self.lock:
            now = time.time()
            dt = min(0.1, now - self.last_update_time)
            
            if dt > 0:
                # Определяем скорость изменения в зависимости от направления
                # Уменьшаем скорость БЫСТРЕЕ, чем увеличиваем
                if self.target_speed_factor < self.current_speed_factor:
                    # Уменьшение скорости (быстрое)
                    adjustment_rate = Config.SPEED_DECAY_RATE * Config.SPEED_SMOOTHING_FACTOR
                else:
                    # Увеличение скорости (плавное)
                    adjustment_rate = Config.SPEED_SMOOTHING_FACTOR
                
                diff = self.target_speed_factor - self.current_speed_factor
                change = diff * adjustment_rate * min(1.0, dt * 10)
                self.current_speed_factor += change
                
                self.current_speed_factor = max(Config.BASE_SPEED, 
                                               min(Config.MAX_QUEUE_FACTOR, 
                                                   self.current_speed_factor))
                self.last_update_time = now
            
            return self.current_speed_factor

class AudioRequest:
    def __init__(self, request_id, session_id, text, speaker, pitch_adjustment, 
                 volume_adjustment, sample_rate, no_interrupt):
        self.request_id = request_id
        self.session_id = session_id
        self.text = text
        self.speaker = speaker
        self.pitch_adjustment = pitch_adjustment
        self.volume_adjustment = volume_adjustment
        self.sample_rate = sample_rate
        self.no_interrupt = no_interrupt
        self.timestamp = time.time()
        self.duration = None
        self.audio_data = None
    
    def cleanup(self):
        """Очистка данных для освобождения памяти"""
        self.text = None
        self.audio_data = None
        self.duration = None

class Session:
    def __init__(self, session_id):
        self.session_id = session_id
        self.queue = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.lock = Lock()
        self.currently_playing = None
        self.play_until = 0
        
    def add_request(self, audio_request):
        """Добавление запроса в очередь сессии"""
        try:
            self.queue.put(audio_request, block=False)
            return True
        except Exception:
            # Очередь переполнена - удаляем самый старый запрос
            try:
                old_request = self.queue.get_nowait()
                old_request.cleanup()
                del old_request
                self.queue.put(audio_request, block=False)
                print(f"[!] Очередь переполнена, удален старый запрос")
                return True
            except:
                return False
    
    def get_queue_size(self):
        """Получение текущего размера очереди"""
        return self.queue.qsize()
    
    def can_process(self):
        """Проверка, можно ли обработать следующий запрос"""
        with self.lock:
            if self.currently_playing is None:
                return True
            return time.time() >= self.play_until

class QueueManager:
    def __init__(self, audio_processor):
        self.audio_processor = audio_processor
        self.sessions = {}
        self.lock = Lock()
        self.speed_controller = DynamicSpeedController()
        self.processing_thread = Thread(target=self._process_loop, daemon=True)
        self.speed_adjustment_thread = Thread(target=self._speed_adjustment_loop, daemon=True)
        self.processing_thread.start()
        self.speed_adjustment_thread.start()
        
    def _get_or_create_session(self, session_id):
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = Session(session_id)
            return self.sessions[session_id]
    
    def _get_total_queue_size(self):
        """Получение общего размера всех очередей"""
        total = 0
        with self.lock:
            for session in self.sessions.values():
                total += session.get_queue_size()
        return total
    
    def add_request(self, session_id, text, speaker, pitch, volume, sample_rate, no_interrupt):
        request_id = str(uuid.uuid4())
        audio_request = AudioRequest(
            request_id, session_id, text, speaker, pitch, volume, sample_rate, no_interrupt
        )
        
        session = self._get_or_create_session(session_id)
        if session.add_request(audio_request):
            return request_id
        return None
        
    def _speed_adjustment_loop(self):
        """Цикл обновления динамической скорости"""
        while True:
            try:
                total_queue_size = self._get_total_queue_size()
                self.speed_controller.update_queue_size(total_queue_size, Config.MAX_QUEUE_SIZE)
                time.sleep(Config.SPEED_ADJUSTMENT_INTERVAL)
            except Exception as e:
                print(f"[X] Ошибка в speed adjustment loop: {e}")
                time.sleep(0.1)
    
    def _return_result(self, audio_request):
        """Отправка результата"""
        self._store_result(audio_request)
        
        # Отложенная очистка
        def delayed_cleanup():
            time.sleep(5)
            audio_request.cleanup()
        
        cleanup_thread = Thread(target=delayed_cleanup, daemon=True)
        cleanup_thread.start()

    def _process_loop(self):
        """Основной цикл обработки очередей"""
        while True:
            try:
                # Получаем текущую динамическую скорость
                dynamic_speed = self.speed_controller.get_current_speed()
                
                with self.lock:
                    sessions_to_check = list(self.sessions.items())
                
                for session_id, session in sessions_to_check:
                    if session.can_process():
                        try:
                            audio_request = session.queue.get_nowait()
                            
                            # Синтезируем аудио с динамической скоростью
                            audio_data = self.audio_processor.synthesize(
                                audio_request.text,
                                audio_request.speaker,
                                dynamic_speed,
                                audio_request.pitch_adjustment,
                                audio_request.volume_adjustment,
                                audio_request.sample_rate
                            )
                            
                            samples = len(audio_data) // 2
                            duration = samples / audio_request.sample_rate
                            
                            audio_request.audio_data = audio_data
                            audio_request.duration = duration
                            
                            # Обновляем состояние сессии
                            with session.lock:
                                if session.currently_playing:
                                    session.currently_playing.cleanup()
                                session.currently_playing = audio_request
                                if audio_request.no_interrupt:
                                    session.play_until = time.time() + duration
                                else:
                                    session.play_until = time.time()
                            
                            # Отправляем результат
                            self._return_result(audio_request)
                            
                        except Empty:
                            pass
                        except Exception as e:
                            print(f"[X] Ошибка обработки запроса: {e}")
                            traceback.print_exc()
                            with session.lock:
                                if session.currently_playing:
                                    session.currently_playing.cleanup()
                                session.currently_playing = None
                                session.play_until = time.time()
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[X] Ошибка в processing loop: {e}")
                time.sleep(0.1)
    
    def _store_result(self, audio_request):
        """Хранение результата (должен быть переопределен)"""
        pass

# ==================== HTTP СЕРВЕР ====================

class TTServer:
    def __init__(self):
        self.app = Bottle()
        self.model = None
        self.cpu_monitor = None
        self.audio_processor = None
        self.queue_manager = None
        self.pending_results = {}
        self.result_lock = Lock()
        self.setup_routes()
        
    def load_model(self):
        if not os.path.exists(Config.MODEL_PATH):
            print(f"\n[X] Модель не найдена: {Config.MODEL_PATH}")
            print(f"[!] Пожалуйста, скачайте модель с https://models.silero.ai/models/tts/ru/v5_5_ru.pt")
            sys.exit(1)
        
        print(f"[*] Загрузка модели... ({Config.DEVICE})")
        package = torch.package.PackageImporter(Config.MODEL_PATH)
        self.model = package.load_pickle("tts_models", "model")
        self.model.to(Config.DEVICE)
        
        # Инициализируем монитор CPU
        self.cpu_monitor = CPUMonitor()
        
        self.audio_processor = AudioProcessor(self.model, Config.DEVICE, self.cpu_monitor)
        self.queue_manager = QueueManager(self.audio_processor)
        self.queue_manager._store_result = self._store_result
        print("[V] Модель загружена")
    
    def _store_result(self, audio_request):
        with self.result_lock:
            self.pending_results[audio_request.request_id] = audio_request.audio_data
    
    def setup_routes(self):
        
        @self.app.route('/voice/speakers', method='GET')
        def get_speakers():
            response.content_type = 'application/json'
            # Возвращаем список голосов с их настройками
            speakers_with_settings = []
            for speaker in SPEAKERS:
                speaker_copy = speaker.copy()
                if speaker["name"] in VOICE_SETTINGS:
                    speaker_copy["settings"] = VOICE_SETTINGS[speaker["name"]]
                speakers_with_settings.append(speaker_copy)
            return {"vits": speakers_with_settings}
        
        @self.app.route('/voice/vits', method='GET')
        def synthesize():
            text = request.query.text or ""
            speaker_id = int(request.query.id or 0)
            speed = float(request.query.speed or 1.0)
            pitch = float(request.query.pitch or 0)
            volume = float(request.query.volume or 0)
            no_interrupt = request.query.no_interrupt != "false"
            sample_rate = Config.SAMPLE_RATE
            
            if not text:
                return {"error": "Text is required"}, 400
            
            text_length = len(text)
            
            if text_length > Config.MAX_TEXT_LENGTH:
                text = text[:Config.MAX_TEXT_LENGTH]
                print(f"[WARNING] Text too long: {text_length} chars (text truncated to {Config.MAX_TEXT_LENGTH} chars)")
            
            print(f"[DEBUG] Text length = {min(text_length, Config.MAX_TEXT_LENGTH)}")
            
            # Получаем имя спикера
            if 0 <= speaker_id < len(SPEAKERS):
                speaker = SPEAKERS[speaker_id]["name"]
            else:
                speaker = "aidar"
            
            # Добавляем в очередь
            request_id = self.queue_manager.add_request(
                "default", text, speaker, pitch, volume, sample_rate, no_interrupt
            )
            
            if not request_id:
                return {"error": "Queue is full, please try again later"}, 503
            
            # Ожидание результата
            timeout = Config.REQUEST_TIMEOUT
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                with self.result_lock:
                    if request_id in self.pending_results:
                        audio_data = self.pending_results.pop(request_id)
                        response.content_type = 'audio/wav'
                        return audio_data
                time.sleep(0.01)
            
            return {"error": "Request timeout"}, 408
        
    def run(self):
        self.load_model()
        print('=' * 70)
        print(f" Silero TTS Server")
        print(f" Device: {Config.DEVICE}")
        print(f" URL: http://{Config.HOST}:{Config.PORT}")
        print('=' * 70)
        print(f"\nCPU thresholds: {Config.CPU_HIGH_THRESHOLD}% (high), {Config.CPU_CRITICAL_THRESHOLD}% (critical)")
        print("\nEndpoints:")
        print("  GET  /voice/speakers        - Get speakers list")
        print("  GET  /voice/vits            - Synthesize speech")
        print('=' * 70)
        
        run(self.app, host=Config.HOST, port=Config.PORT, quiet=False)

if __name__ == "__main__":
    server = TTServer()
    server.run()