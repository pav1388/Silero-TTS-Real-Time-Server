# cpu_monitor.py
# pav13

import threading
import time
import logging
import psutil

logger = logging.getLogger(__name__)

class CPUMonitor:
    """Мониторинг загрузки CPU для динамического изменения качества"""
    def __init__(self, idle_timeout=30.0, monitor_interval=1.0, sample_duration=0.07,
                            high_thresh=85.0, critical_thresh=95.0, quality_levels = None):
        if quality_levels:
            self.quality_levels = quality_levels
        else:
            self.quality_levels = [
                {"sample_rate": 8000,  "put_accent": False, "put_yo": False,
                        "put_stress_homo": False, "put_yo_homo": False, "name": "LOWEST"},
                {"sample_rate": 8000,  "put_accent": True,  "put_yo": False,
                        "put_stress_homo": True,  "put_yo_homo": False, "name": "LOW"},
                {"sample_rate": 24000, "put_accent": False, "put_yo": False,
                        "put_stress_homo": False, "put_yo_homo": False, "name": "MED-LOW"},
                {"sample_rate": 24000, "put_accent": True,  "put_yo": True,
                        "put_stress_homo": True,  "put_yo_homo": True,  "name": "MED"},
                {"sample_rate": 48000, "put_accent": True,  "put_yo": False,
                        "put_stress_homo": True,  "put_yo_homo": False, "name": "HIGH"},
                {"sample_rate": 48000, "put_accent": True,  "put_yo": True,
                        "put_stress_homo": True,  "put_yo_homo": True,  "name": "MAX"}
            ]
        
        self.max_level = len(self.quality_levels) - 1
        self.current_quality_level = self.max_level
        
        self.idle_timeout = idle_timeout
        self.monitor_interval = monitor_interval
        self.sample_duration = sample_duration
        self.high_thresh = high_thresh
        self.critical_thresh = critical_thresh
        
        self.current_load = 0.0
        self.last_change_time = 0
        self.lock = threading.Lock()
        self.running = False
        self.load_history = []
        self.max_history_size = 3
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
            logger.debug("CPU monitoring ON")

    def stop(self):
        with self.lock:
            if not self.running: return
            self.running = False
            logger.debug("CPU monitoring OFF")

    def record_activity(self):
        self.last_activity_time = time.time()
        if not self.running: self.start()

    def _check_idle_and_stop(self):
        if time.time() - self.last_activity_time >= self.idle_timeout:
            self.stop()
            return True
        return False

    def _get_cpu_load(self) -> float:
        try: return psutil.cpu_percent(interval=self.sample_duration)
        except: return 0.0

    def _add_to_history(self, value: float):
        self.load_history.append(value)
        if len(self.load_history) > self.max_history_size: self.load_history.pop(0)

    def _get_average_load(self) -> float:
        return sum(self.load_history) / len(self.load_history) if self.load_history else 0.0

    def _calculate_target_quality(self, avg_load: float) -> int:
        if avg_load >= self.critical_thresh: return 0
        if avg_load >= self.high_thresh:
            load_ratio = (avg_load - self.high_thresh) / (self.critical_thresh - self.high_thresh)
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
                        logger.debug(f"Quality {'UP' if self.current_quality_level > old_level else 'DOWN'} to Lvl {self.current_quality_level} LOAD {avg_load:.1f}%")
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(5)

    def get_current_quality_config(self) -> dict:
        with self.lock: 
            return self.quality_levels[self.current_quality_level].copy()

    def get_cpu_load(self) -> float:
        with self.lock: return self.current_load