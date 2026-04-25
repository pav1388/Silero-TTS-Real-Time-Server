import io
import os
import sys
import time
import random
import struct
from collections import deque
from functools import lru_cache
from threading import Lock, Event, Thread
from queue import Queue
from urllib.parse import unquote

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from num2words import num2words

# ==================== КОНФИГУРАЦИЯ ====================

class Config:
    """Глобальные настройки приложения"""
    MODEL_PATH = "models/v5_5_ru.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAMPLE_RATE_DEFAULT = 48000
    HOST = "127.0.0.1" # все интерфейсы "0.0.0.0"
    PORT = 5000
    
    @classmethod
    def get_max_workers(cls):
        if hasattr(os, 'sched_getaffinity'):
            cpu_cores = len(os.sched_getaffinity(0))
        elif os.name == 'nt':
            cpu_cores = int(os.environ.get('NUMBER_OF_PROCESSORS', 1))
        else:
            cpu_cores = os.cpu_count() or 1
        
        if cls.DEVICE.type == "cuda":
            return min(16, cpu_cores * 4)
        return cpu_cores 


class AudioPauses:
    """Настройки пауз в миллисекундах"""
    SENTENCE = 320
    COMMA = 220
    SENTENCE_END = 400


class SpeedConfig:
    """Конфигурация автоскорости"""
    BASE_SPEED = 1.15
    MAX_SPEED = 2.0
    MIN_SPEED = 1.15
    AGGRESSIVENESS = 0.8
    SMOOTHING = 0.3
    PREDICTION_CONFIDENCE = 0.7
    MIN_COMPLETION_RATIO = 0.8
    TARGET_COMPLETION_RATIO = 0.95
    EMERGENCY_SPEED_BOOST = 0.4


# ==================== ДАННЫЕ ГОЛОСОВ ====================

SPEAKERS = [
    {"id": 0, "name": "aidar", "style": "male", "lang": ["ru"]},
    {"id": 1, "name": "baya", "style": "female", "lang": ["ru"]},
    {"id": 2, "name": "kseniya", "style": "female", "lang": ["ru"]},
    {"id": 3, "name": "xenia", "style": "female", "lang": ["ru"]},
    {"id": 4, "name": "eugene", "style": "male", "lang": ["ru"]},
]

SPEAKER_SETTINGS = {
    "aidar": {"volume_boost": 3, "pitch": "high", "base_speed": 1.1},
    "eugene": {"volume_boost": 0.5, "pitch": "low", "base_speed": 0.9},
    "baya": {"volume_boost": 0, "pitch": "low", "base_speed": 1.0},
    "kseniya": {"volume_boost": 0, "pitch": "low", "base_speed": 1.0},
    "xenia": {"volume_boost": 1, "pitch": "medium", "base_speed": 0.95}
}

# ==================== УТИЛИТЫ ====================

def uuid() -> str:
    """Генератор уникальных ID"""
    return f"{time.time_ns():x}{random.getrandbits(32):08x}"

def is_digit_string(s: str) -> bool:
    """Проверка, состоит ли строка только из цифр (без re)"""
    if not s:
        return False
    for ch in s:
        if ch not in '0123456789':
            return False
    return True

def is_valid_volume(volume_str: str) -> bool:
    """Проверка формата громкости: +-число (без re)"""
    if not volume_str:
        return False
    
    # Удаляем пробелы
    volume_str = volume_str.replace(" ", "")
    
    if not volume_str:
        return False
    
    # Проверяем первый символ
    start_idx = 0
    if volume_str[0] in '+-':
        start_idx = 1
        if len(volume_str) == 1:
            return False
    
    # Проверяем, что все оставшиеся символы - цифры
    for i in range(start_idx, len(volume_str)):
        if volume_str[i] not in '0123456789':
            return False
    
    return True

class TextProcessor:
    """Обработка текста за один проход: числа, транслитерация, пунктуация"""
    
    def __init__(self):
        self._init_trie()
        self._init_punctuation_config()
        
        # Разрешенные символы для модели (русские буквы + базовая пунктуация)
        self.allowed_chars = set("_~абвгдеёжзийклмнопрстуфхцчшщъыьэюя +.,!?…:;–")
    
    def _init_trie(self):
        """Построение trie-дерева для транслитерации"""
        # Транслитерация латиницы
        TRANSLIT_MAP = {
            'ough': 'о', 'augh': 'о', 'eigh': 'эй', 'tion': 'шн', 'shch': 'щ',
            'tch': 'ч', 'sch': 'ск', 'scr': 'скр',
            'thr': 'зр', 'squ': 'скв', 'ear': 'ир', 'air': 'эр', 'are': 'эр',
            'the': 'зэ', 'and': 'энд',
            'ea': 'и', 'ee': 'и', 'oo': 'у', 'ai': 'эй', 'ay': 'эй',
            'ei': 'эй', 'ey': 'эй', 'oi': 'ой', 'oy': 'ой',
            'ou': 'ау', 'ow': 'ау', 'au': 'о', 'aw': 'о',
            'ie': 'и', 'ui': 'у', 'ue': 'ю', 'uo': 'уо',
            'eu': 'ю', 'ew': 'ю', 'oa': 'о', 'oe': 'о',
            'sh': 'ш', 'ch': 'ч', 'zh': 'ж', 'th': 'з',
            'kh': 'х', 'ts': 'ц', 'ph': 'ф',
            'wh': 'в', 'gh': 'г', 'qu': 'кв', 'gu': 'г', 'dg': 'дж',
            'ce': 'це', 'ci': 'си', 'cy': 'си', 'ck': 'к',
            'ge': 'дж', 'gi': 'джи', 'gy': 'джи', 'er': 'эр',
            'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е',
            'f': 'ф', 'g': 'г', 'h': 'х', 'i': 'и',
            'j': 'дж', 'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н',
            'o': 'о', 'p': 'п', 'q': 'к', 'r': 'р',
            's': 'с', 't': 'т', 'u': 'у', 'v': 'в', 'w': 'в',
            'x': 'кс', 'y': 'й', 'z': 'з',
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
        """Конфигурация пунктуации с паузами"""
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
        """
        Полная обработка текста за один проход
        
        Args:
            text: Входной текст
            add_final_pause: Добавлять финальную паузу в конце
        
        Returns:
            Обработанный текст с SSML разметкой
        """
        if not text:
            return ""
        
        # Начальная подготовка
        text = unquote(text).lower()
        result_parts = []
        i = 0
        n = len(text)
        
        # Состояния
        ends_with_sentence = False
        last_was_space = False
        
        while i < n:
            ch = text[i]
            
            # 1. ОБРАБОТКА ЧИСЕЛ
            if ch.isdigit():
                i, number_result = self._process_number(text, i)
                result_parts.append(number_result)
                ends_with_sentence = False
                last_was_space = False
                continue
            
            # 2. ТРАНСЛИТЕРАЦИЯ
            if ch.isalpha():
                i, translit_result = self._process_transliteration(text, i)
                if translit_result:
                    result_parts.append(translit_result)
                    ends_with_sentence = False
                    last_was_space = False
                    continue
            
            # 3. ПУНКТУАЦИЯ С ПАУЗАМИ
            if ch in self.punctuation_config:
                config = self.punctuation_config[ch]
                
                # Удаляем пробел перед пунктуацией (если есть)
                if result_parts and result_parts[-1] == ' ':
                    result_parts.pop()
                
                result_parts.append(ch)
                result_parts.append(f'<break time="{config["pause"]}ms"/> ')
                ends_with_sentence = config['end_sentence']
                last_was_space = True
                i += 1
                continue
            
            # 4. ПРОБЕЛЫ (нормализация)
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
            
            # 6. ОСТАЛЬНЫЕ СИМВОЛЫ - заменяем на пробел
            if not last_was_space:
                result_parts.append(' ')
                last_was_space = True
            i += 1
        
        # Финальная обработка
        result = ''.join(result_parts).strip()
        
        # Добавляем финальную паузу если нужно
        if add_final_pause and ends_with_sentence:
            result += f'<break time="{AudioPauses.SENTENCE_END}ms"/>'
        
        return result
    
    def _process_number(self, text: str, start: int) -> tuple:
        """
        Обработка числа и возврат его текстового представления
        
        Returns:
            (new_position, text_result)
        """
        i = start
        n = len(text)
        
        # Читаем первую часть числа
        j = i
        while j < n and text[j].isdigit():
            j += 1
        
        if j == i:  # Не число
            return i + 1, text[i]
        
        num1 = text[i:j]
        
        # Проверка на десятичную дробь
        if j < n and text[j] in '.,' and j + 1 < n and text[j + 1].isdigit():
            k = j + 1
            while k < n and text[k].isdigit():
                k += 1
            num2 = text[j + 1:k]
            
            result = f"{num_to_words(num1)} точка {num_to_words(num2)}"
            return k, result
        
        # Проверка на обычную дробь
        if j < n and text[j] == '/' and j + 1 < n and text[j + 1].isdigit():
            k = j + 1
            while k < n and text[k].isdigit():
                k += 1
            num2 = text[j + 1:k]
            
            result = f"{num_to_words(num1)} дробь {num_to_words(num2)}"
            return k, result
        
        # Обычное число
        return j, num_to_words(num1)
    
    def _process_transliteration(self, text: str, pos: int) -> tuple:
        """Поиск и замена транслитерации"""
        node = self.translit_trie
        best_match = None
        best_pos = pos
        j = pos
        
        # Ищем самое длинное совпадение
        while j < len(text) and text[j] in node:
            node = node[text[j]]
            j += 1
            if '_' in node:
                best_match = node['_']
                best_pos = j
        
        if best_match:
            return best_pos, best_match
        
        # Если не нашли транслитерацию, возвращаем оригинальный символ
        return pos + 1, text[pos] if text[pos] in self.allowed_chars else " "
    

class AudioProcessor:
    """Обработка аудио с интегрированным TextProcessor"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.text_processor = TextProcessor()
    
    def synthesize(self, text: str, speaker: str, speed: str, pitch: str, 
                   volume: str, autospeed: bool, sample_rate: int) -> bytes:
        """Основной метод синтеза речи - возвращает WAV байты"""
        try:
            # Валидация и подготовка параметров
            valid_speaker = self._validate_speaker(speaker)
            rate = self._calculate_speed(speed, valid_speaker, autospeed)
            normalized_pitch = self._normalize_pitch(pitch, valid_speaker)
            
            # Обработка текста
            processed_text = self.text_processor.process_text(text)
            
            if not processed_text:
                # Возвращаем тишину как WAV байты
                silent_audio = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
                return self._apply_volume(silent_audio, speaker, volume, sample_rate)
            
            # Создание SSML
            ssml = f'<speak><prosody rate="{rate}" pitch="{normalized_pitch}">{processed_text}</prosody></speak>'
            
            # Генерация аудио
            audio_np = self._generate_audio(ssml, valid_speaker, sample_rate)
            
            # Постобработка громкости и возврат WAV байтов
            return self._apply_volume(audio_np, valid_speaker, volume, sample_rate)
            
        except Exception as e:
            print(f"[X] Ошибка синтеза: {e}")
            # Возвращаем тишину при ошибке
            silent_audio = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
            return self._apply_volume(silent_audio, speaker, volume, sample_rate)
    
    def _validate_speaker(self, speaker: str) -> str:
        """Валидация имени спикера"""
        for s in SPEAKERS:
            if s["name"] == speaker:
                return speaker
        return "aidar"
    
    def _calculate_speed(self, speed_param: str, speaker: str, autospeed: bool) -> str:
        """Расчет финальной скорости воспроизведения"""
        base_speed = SPEAKER_SETTINGS.get(speaker, {}).get("base_speed", 1.0)
        
        try:
            user_speed = max(0.5, min(2.0, float(speed_param)))
        except:
            user_speed = 1.0
        
        if autospeed:
            auto_multiplier = speed_manager.get_autospeed()
            final_speed = base_speed * user_speed * auto_multiplier
        else:
            final_speed = base_speed * user_speed
        
        return f"{int(final_speed * 100)}%"
    
    def _normalize_pitch(self, pitch: str, speaker: str) -> str:
        """Нормализация высоты тона"""
        valid_pitches = ["x-low", "low", "medium", "high", "x-high"]
        
        if pitch and pitch.lower() in valid_pitches:
            return pitch.lower()
        
        return SPEAKER_SETTINGS.get(speaker, {}).get("pitch", "medium")
    
    def _generate_audio(self, ssml: str, speaker: str, sample_rate: int) -> np.ndarray:
        """Генерация аудио через модель"""
        audio = self.model.apply_tts(
            ssml_text=ssml,
            speaker=speaker,
            sample_rate=sample_rate,
            put_accent=True,
            put_yo=True
        )
        
        if hasattr(audio, 'cpu'):
            return audio.cpu().numpy()
        return np.array(audio, dtype=np.float32)
    
    def _apply_volume(self, audio_np: np.ndarray, speaker: str, 
                      volume_param: str, sample_rate: int) -> bytes:
        """Применение настроек громкости и возврат WAV байтов"""
        speaker_boost = SPEAKER_SETTINGS.get(speaker, {}).get("volume_boost", 0)
        user_volume = self._parse_volume(volume_param)
        total_boost_db = speaker_boost + user_volume
        
        if total_boost_db != 0:
            # Конвертируем dB в линейный коэффициент
            volume_factor = 10 ** (total_boost_db / 20.0)
            audio_np = np.clip(audio_np * volume_factor, -1.0, 1.0)
        
        # Конвертируем в int16
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        # Возвращаем WAV байты (ручное формирование без wave)
        return self._numpy_to_wav_bytes(audio_int16, sample_rate)
    
    def _numpy_to_wav_bytes(self, audio_int16: np.ndarray, sample_rate: int) -> bytes:
        """Конвертация numpy массива в WAV байты"""
        buffer = io.BytesIO()
        
        # Параметры WAV файла
        channels = 1  # Моно
        bits_per_sample = 16
        bytes_per_sample = bits_per_sample // 8
        byte_rate = sample_rate * channels * bytes_per_sample
        block_align = channels * bytes_per_sample
        data_size = len(audio_int16) * bytes_per_sample
        file_size = 36 + data_size  # 44 байта заголовок - 8 = 36
        
        # RIFF заголовок
        buffer.write(b'RIFF')
        buffer.write(struct.pack('<I', file_size))
        buffer.write(b'WAVE')
        
        # fmt chunk
        buffer.write(b'fmt ')
        buffer.write(struct.pack('<I', 16))  # размер fmt чанка
        buffer.write(struct.pack('<H', 1))   # аудио формат (PCM)
        buffer.write(struct.pack('<H', channels))
        buffer.write(struct.pack('<I', sample_rate))
        buffer.write(struct.pack('<I', byte_rate))
        buffer.write(struct.pack('<H', block_align))
        buffer.write(struct.pack('<H', bits_per_sample))
        
        # data chunk
        buffer.write(b'data')
        buffer.write(struct.pack('<I', data_size))
        buffer.write(audio_int16.tobytes())
        
        return buffer.getvalue()
    
    def _parse_volume(self, volume_param: str) -> int:
        """Парсинг параметра громкости"""
        if not volume_param:
            return 0
        
        volume_str = str(volume_param).replace(" ", "")
        
        # Простая проверка формата без re
        if not volume_str:
            return 0
        
        start_idx = 0
        if volume_str[0] in '+-':
            start_idx = 1
            if len(volume_str) == 1:
                print(f"[!] Неверный формат volume: '{volume_param}'. Используйте числа: +6, -3, 10")
                return 0
        
        # Проверяем, что все символы - цифры
        for i in range(start_idx, len(volume_str)):
            if volume_str[i] not in '0123456789':
                print(f"[!] Неверный формат volume: '{volume_param}'. Используйте числа: +6, -3, 10")
                return 0
        
        # Парсим число
        db_value = int(volume_str)
        return max(-30, min(30, db_value))


# ==================== УПРАВЛЕНИЕ СКОРОСТЬЮ ====================

class SpeedManager:
    """Управление скоростью воспроизведения"""
    
    def __init__(self, config_class):
        self.config = {
            "base_speed": config_class.BASE_SPEED,
            "max_speed": config_class.MAX_SPEED,
            "min_speed": config_class.MIN_SPEED,
            "aggressiveness": config_class.AGGRESSIVENESS,
            "smoothing": config_class.SMOOTHING,
            "prediction_confidence": config_class.PREDICTION_CONFIDENCE,
            "min_completion_ratio": config_class.MIN_COMPLETION_RATIO,
            "target_completion_ratio": config_class.TARGET_COMPLETION_RATIO,
            "emergency_speed_boost": config_class.EMERGENCY_SPEED_BOOST,
        }
        
        self.current_dialog_id = None
        self.lock = Lock()
        
        # Очереди и история
        self.playback_queue = deque(maxlen=20)
        self.request_history = deque(maxlen=15)
        self.completion_ratios = deque(maxlen=10)
        
        # Текущее состояние
        self.last_audio_duration = 0
        self.last_request_time = 0
        self.current_speed = self.config["base_speed"]
        self.target_speed = self.config["base_speed"]
        
        # Прогнозы
        self.predicted_interval = 2.0
        self.avg_audio_duration = 3.0
    
    def start_dialog(self) -> str:
        """Начать новый диалог"""
        with self.lock:
            self.current_dialog_id = uuid()
            return self.current_dialog_id
    
    def get_current_dialog(self) -> str:
        """Получить ID текущего диалога"""
        with self.lock:
            return self.current_dialog_id
    
    def update_config(self, new_config: dict):
        """Обновить конфигурацию"""
        with self.lock:
            self.config.update(new_config)
            print(f"[V] Обновлена конфигурация: {new_config}")
    
    def record_request(self, audio_duration_ms: int = 0, speaker: str = None):
        """Записать запрос и скорректировать скорость"""
        with self.lock:
            current_time = time.time()
            
            if audio_duration_ms > 0:
                self.last_audio_duration = audio_duration_ms / 1000.0
                self._update_average_duration(self.last_audio_duration)
            
            if self.last_request_time > 0:
                interval = current_time - self.last_request_time
                self.request_history.append(interval)
                self._update_predicted_interval()
            
            self._clean_completed_dialogs(current_time)
            self._add_to_playback_queue(audio_duration_ms, speaker, current_time)
            self._intelligent_speed_adjustment(current_time)
            
            self.last_request_time = current_time
    
    def _clean_completed_dialogs(self, current_time: float):
        """Очистить завершенные диалоги"""
        completed_count = 0
        while self.playback_queue and self.playback_queue[0]["completion_time"] <= current_time:
            completed = self.playback_queue.popleft()
            
            if len(self.playback_queue) > 1:
                actual_completion = completed["completion_time"]
                next_request = self.playback_queue[0]["added_time"]
                ratio = 1.0 if actual_completion <= next_request else 0.0
                self.completion_ratios.append(ratio)
            
            completed_count += 1
        return completed_count
    
    def _add_to_playback_queue(self, duration_ms: int, speaker: str, current_time: float):
        """Добавить диалог в очередь воспроизведения"""
        if duration_ms <= 0:
            return
        
        playback_time = (duration_ms / 1000.0) / self.current_speed
        self.playback_queue.append({
            "completion_time": current_time + playback_time,
            "original_duration": duration_ms / 1000.0,
            "playback_duration": playback_time,
            "speaker": speaker,
            "added_time": current_time
        })
    
    def _update_average_duration(self, new_duration: float):
        """Обновить среднюю длительность аудио"""
        alpha = 0.3
        if self.avg_audio_duration == 0:
            self.avg_audio_duration = new_duration
        else:
            self.avg_audio_duration = (alpha * new_duration + 
                                      (1 - alpha) * self.avg_audio_duration)
    
    def _update_predicted_interval(self):
        """Обновить прогнозируемый интервал"""
        if self.request_history:
            intervals = sorted(self.request_history)
            median_interval = intervals[len(intervals) // 2]
            alpha = self.config["prediction_confidence"]
            self.predicted_interval = (alpha * median_interval + 
                                      (1 - alpha) * self.predicted_interval)
    
    def _calculate_required_speed(self, current_time: float) -> float:
        """Рассчитать требуемую скорость"""
        if not self.playback_queue:
            return self.config["base_speed"]
        
        current_dialog = self.playback_queue[0]
        time_remaining = current_dialog["completion_time"] - current_time
        time_until_next = self.predicted_interval
        original_duration = current_dialog["original_duration"]
        
        if time_until_next <= 0:
            required_speed = original_duration / 0.1
        else:
            required_speed = original_duration / time_until_next
        
        return max(self.config["min_speed"], 
                  min(self.config["max_speed"], required_speed))
    
    def _calculate_queue_pressure(self) -> float:
        """Рассчитать давление очереди"""
        if not self.playback_queue:
            return 0.0
        queue_size = len(self.playback_queue)
        return min(1.0, (queue_size - 1) * 0.3)
    
    def _calculate_completion_risk(self, current_time: float) -> float:
        """Рассчитать риск невыполнения"""
        if len(self.playback_queue) < 2:
            return 0.0
        
        current_dialog = self.playback_queue[0]
        time_to_complete = current_dialog["completion_time"] - current_time
        original_duration = current_dialog["original_duration"]
        
        if time_to_complete > 0:
            risk = min(1.0, time_to_complete / original_duration)
        else:
            risk = 0.0
        
        return risk
    
    def _intelligent_speed_adjustment(self, current_time: float):
        """Интеллектуальная корректировка скорости"""
        if not self.playback_queue:
            self.target_speed = self.config["base_speed"]
            return
        
        required_speed = self._calculate_required_speed(current_time)
        queue_pressure = self._calculate_queue_pressure()
        completion_risk = self._calculate_completion_risk(current_time)
        
        target = required_speed
        
        if queue_pressure > 0:
            target *= (1.0 + queue_pressure * self.config["aggressiveness"])
        
        if completion_risk > 0.5:
            emergency_boost = completion_risk * self.config["emergency_speed_boost"]
            target *= (1.0 + emergency_boost)
            print(f" ЭКСТРЕННОЕ УСКОРЕНИЕ: риск={completion_risk:.2f}, +{emergency_boost:.2f}")
        
        new_target = max(self.config["min_speed"], 
                        min(self.config["max_speed"], target))
        
        # Плавное изменение
        speed_diff = new_target - self.target_speed
        self.target_speed += speed_diff * self.config["smoothing"]
        
        # Логирование
        if abs(speed_diff) > 0.1 or len(self.playback_queue) > 2:
            risk_info = f", риск: {completion_risk:.2f}" if completion_risk > 0.1 else ""
            print(f" АДАПТАЦИЯ: очередь={len(self.playback_queue)}, "
                  f"требуется={required_speed:.2f}x, "
                  f"целевая={self.target_speed:.2f}x{risk_info}")
    
    def _apply_smooth_speed_change(self):
        """Плавное применение изменения скорости"""
        speed_diff = self.target_speed - self.current_speed
        self.current_speed += speed_diff * self.config["smoothing"]
        
        if abs(speed_diff) < 0.01:
            self.current_speed = self.target_speed
    
    def get_autospeed(self) -> float:
        """Получить текущую автоскорость"""
        with self.lock:
            self._apply_smooth_speed_change()
            return self.current_speed
    
    def get_smart_stats(self) -> dict:
        """Получить детальную статистику"""
        with self.lock:
            current_time = time.time()
            queue_size = len(self.playback_queue)
            
            avg_completion = (sum(self.completion_ratios) / len(self.completion_ratios) 
                            if self.completion_ratios else 1.0)
            current_risk = self._calculate_completion_risk(current_time)
            required_speed = self._calculate_required_speed(current_time)
            
            # Определение режима
            if current_risk > 0.7:
                mode = "[!] КРИТИЧЕСКИЙ"
            elif current_risk > 0.3:
                mode = "[!] ВЫСОКИЙ РИСК"
            elif queue_size > 2:
                mode = "[!] ВЫСОКАЯ НАГРУЗКА"
            elif queue_size > 1:
                mode = "[!] СРЕДНЯЯ НАГРУЗКА"
            else:
                mode = "[V] НОРМА"
            
            return {
                "current_speed": round(self.current_speed, 3),
                "target_speed": round(self.target_speed, 3),
                "required_speed": round(required_speed, 3),
                "queue_size": queue_size,
                "queue_pressure": round(self._calculate_queue_pressure(), 3),
                "completion_risk": round(current_risk, 3),
                "avg_completion_ratio": round(avg_completion, 3),
                "predicted_interval": round(self.predicted_interval, 2),
                "avg_audio_duration": round(self.avg_audio_duration, 2),
                "last_audio_duration": round(self.last_audio_duration, 2),
                "mode": mode,
                "efficiency": f"{avg_completion*100:.1f}%",
                "config": self.config
            }
    
    def reset(self):
        """Сбросить всю статистику"""
        with self.lock:
            self.playback_queue.clear()
            self.request_history.clear()
            self.completion_ratios.clear()
            self.last_audio_duration = 0
            self.last_request_time = 0
            self.current_speed = self.config["base_speed"]
            self.target_speed = self.config["base_speed"]
            self.predicted_interval = 2.0
            self.avg_audio_duration = 3.0


# ==================== ОЧЕРЕДЬ ОБРАБОТКИ ====================

class TTSQueue:
    """Очередь задач синтеза речи"""
    
    def __init__(self, workers_count: int, audio_processor: AudioProcessor):
        self.queue = Queue()
        self.workers_count = workers_count
        self.audio_processor = audio_processor
        self.running = True
        self.workers = []
        
        self._start_workers()
    
    def _start_workers(self):
        """Запуск рабочих потоков"""
        for _ in range(self.workers_count):
            worker = Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self):
        """Основной цикл рабочего потока"""
        while self.running:
            try:
                task = self.queue.get(timeout=1)
                if task is None:
                    self.queue.task_done()
                    break
                
                text, speaker, speed, pitch, volume, autospeed, sample_rate, callback, duration_callback = task
                
                # Получаем WAV байты напрямую
                wav_bytes = self.audio_processor.synthesize(
                    text, speaker, speed, pitch, volume, autospeed, sample_rate
                )
                
                if duration_callback and wav_bytes:
                    # Извлекаем длительность из WAV заголовка (быстрый парсинг)
                    duration_ms = self._get_wav_duration_fast(wav_bytes, sample_rate)
                    duration_callback(duration_ms)
                
                callback(wav_bytes)  # Передаем готовые WAV байты
                self.queue.task_done()
                
            except Exception:
                continue
    
    def _get_wav_duration_fast(self, wav_bytes: bytes, sample_rate: int) -> int:
        """Получение длительности WAV без распаковки всего файла"""
        # WAV заголовок имеет фиксированную структуру
        # Размер данных находится по смещению 40 байт
        if len(wav_bytes) >= 44:
            # Читаем 4 байта начиная с 40 позиции (little-endian)
            data_size = struct.unpack('<I', wav_bytes[40:44])[0]
            # Длительность в мс = (размер_данных / 2) / частота * 1000
            # /2 для 16-bit моно
            duration_ms = int((data_size / 2) / sample_rate * 1000)
            return duration_ms
        return 0
    
    def add_task(self, task):
        """Добавить задачу в очередь"""
        self.queue.put(task)
    
    def size(self) -> int:
        """Размер очереди"""
        return self.queue.qsize()
    
    def shutdown(self):
        """Остановка очереди"""
        self.running = False
        for _ in range(self.workers_count):
            self.queue.put(None)


# ==================== FLASK ПРИЛОЖЕНИЕ ====================

app = Flask(__name__)

# Глобальные инициализации
@lru_cache(maxsize=4096)
def num_to_words(num: str) -> str:
    """Конвертация числа в слова"""
    if len(num) > 9:
        return num
    return num2words(num, lang='ru')

def load_model():
    """Загрузка модели TTS"""
    if not os.path.exists(Config.MODEL_PATH):
        print(f"\n[X] Модель не найдена: {Config.MODEL_PATH}")
        print(" Скачайте файл в папку models: https://models.silero.ai/models/tts/ru/v5_5_ru.pt")
        input(" Нажмите Enter для выхода...")
        sys.exit(1)
    
    package = torch.package.PackageImporter(Config.MODEL_PATH)
    model = package.load_pickle("tts_models", "model")
    model.to(Config.DEVICE)
    return model

# Инициализация компонентов
model = load_model()
audio_processor = AudioProcessor(model, Config.DEVICE)
speed_manager = SpeedManager(SpeedConfig)
tts_queue = TTSQueue(Config.get_max_workers(), audio_processor)


# ==================== API ЭНДПОИНТЫ ====================

@app.route("/voice/speakers", methods=["GET"])
def get_speakers():
    """Получить список доступных голосов"""
    return jsonify({"vits": SPEAKERS})


@app.route("/voice/vits", methods=["GET"])
def synthesize():
    """Основной эндпоинт синтеза речи"""
    start_time = time.time()
    
    try:
        # Получение параметров
        text = request.args.get("text", "").strip()
        speaker_id = int(request.args.get("id", 0))
        speed_param = request.args.get("speed", "1.0")
        pitch_param = request.args.get("pitch", "")
        volume_param = request.args.get("volume", "")
        autospeed_param = request.args.get("autospeed", "false")
        sample_rate_param = request.args.get("sample_rate", str(Config.SAMPLE_RATE_DEFAULT))
        
        # Валидация
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        if not (0 <= speaker_id < len(SPEAKERS)):
            return jsonify({"error": "Invalid speaker ID"}), 400
        
        speaker_name = SPEAKERS[speaker_id]["name"]
        autospeed_enabled = autospeed_param.lower() in ['true', '1', 'yes', 'on']
        sample_rate = int(sample_rate_param)
        
        if sample_rate not in {8000, 24000, 48000}:
            sample_rate = Config.SAMPLE_RATE_DEFAULT
        
        # Начало диалога
        speed_manager.start_dialog()
        
        # Синтез через очередь
        result_wav_bytes = None
        audio_duration = 0
        event = Event()
        
        def duration_callback(duration_ms):
            nonlocal audio_duration
            audio_duration = duration_ms
            speed_manager.record_request(audio_duration, speaker_name)
        
        def callback(wav_bytes):
            nonlocal result_wav_bytes
            result_wav_bytes = wav_bytes
            event.set()
        
        task = (text, speaker_name, speed_param, pitch_param, volume_param, 
                autospeed_enabled, sample_rate, callback, duration_callback)
        tts_queue.add_task(task)
        
        if not event.wait(timeout=15.0):
            return jsonify({"error": "Timeout"}), 408
        
        # Отправляем WAV байты
        response = send_file(
            io.BytesIO(result_wav_bytes),
            mimetype="audio/wav",
            as_attachment=False
        )
        
        # Логирование
        total_time = time.time() - start_time
        current_autospeed = speed_manager.get_autospeed()
        auto_status = f"AUTO({current_autospeed:.2f}x)" if autospeed_enabled else "MANUAL"
        
        stats = speed_manager.get_smart_stats()
        mode_info = f" [{stats['mode']}]" if autospeed_enabled and stats["mode"] != "[V] НОРМА" else ""
        
        print(f"[V] {total_time:.2f}s: {speaker_name}, speed={speed_param}({auto_status}){mode_info}, {len(text)} chars")
        print(text) # для отладки
        
        return response
    
    except Exception as e:
        print(f"[X] Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/voice/speaker_settings", methods=["GET"])
def get_speaker_settings():
    """Получить настройки голосов"""
    return jsonify({"speaker_settings": SPEAKER_SETTINGS})


@app.route("/voice/autospeed/config", methods=["GET", "POST"])
def autospeed_config():
    """Управление конфигурацией автоскорости"""
    if request.method == "GET":
        return jsonify({
            "current_config": speed_manager.config,
            "description": "Текущая конфигурация автоскорости"
        })
    
    elif request.method == "POST":
        try:
            new_config = request.get_json()
            if not new_config:
                return jsonify({"error": "No configuration provided"}), 400
            
            speed_manager.update_config(new_config)
            return jsonify({
                "status": "config_updated",
                "new_config": speed_manager.get_smart_stats()["config"]
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/voice/autospeed/status", methods=["GET"])
def autospeed_status():
    """Получить статус автоскорости"""
    stats = speed_manager.get_smart_stats()
    return jsonify({
        "current_speed": stats["current_speed"],
        "target_speed": stats["target_speed"],
        "required_speed": stats["required_speed"],
        "intelligent_statistics": stats,
        "description": f"Режим: {stats['mode']}, эффективность: {stats['efficiency']}, скорость: {stats['current_speed']}x"
    })


@app.route("/voice/autospeed/reset", methods=["POST"])
def reset_autospeed():
    """Сбросить автоскорость"""
    speed_manager.reset()
    return jsonify({"status": "autospeed_reset", "current_speed": SpeedConfig.BASE_SPEED})


@app.route("/voice/interrupt", methods=["POST"])
def interrupt_dialog():
    """Прервать текущий диалог"""
    speed_manager.start_dialog()
    return jsonify({"status": "interrupted"})


@app.route("/health", methods=["GET"])
def health_check():
    """Проверка здоровья сервера"""
    stats = speed_manager.get_smart_stats()
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "workers": Config.get_max_workers(),
        "queue_size": tts_queue.size(),
        "current_dialog": speed_manager.get_current_dialog(),
        "current_autospeed": stats["current_speed"],
        "mode": stats["mode"],
        "efficiency": stats["efficiency"]
    })


# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    print('=' * 70)
    print(f" Silero TTS сервер (модель {Config.MODEL_PATH})")
    print(f" Устройство: {Config.DEVICE}", end="")
    
    if Config.DEVICE.type == 'cuda':
        print(f" - {torch.cuda.get_device_name(0)}")
        print(f" CUDA version: {torch.version.cuda}")
    else:
        print()
    
    print(f" Потоков: {Config.get_max_workers()}")
    
    speakers_list = ", ".join([f"{s['name']} ({s['style']})" for s in SPEAKERS])
    print(f" Доступные голоса ({len(SPEAKERS)}): {speakers_list}")
    
    print("\n API МОНИТОРИНГА:")
    print("   GET  /voice/autospeed/status - детальная статистика и метрики")
    print("   GET  /voice/autospeed/config - текущая конфигурация")
    print("   POST /voice/autospeed/config - изменить настройки")
    print('=' * 70)
    
    try:
        app.run(host=Config.HOST, port=Config.PORT, threaded=True)
    finally:
        print("\n[V] Остановка сервера...")
        tts_queue.shutdown()
        print("[V] Ресурсы освобождены")