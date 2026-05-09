# cpu_monitor.py
# pav13

import threading
import time
import logging
import os
import platform

logger = logging.getLogger(__name__)

class CPUMonitor:
    def __init__(self, idle_timeout=30.0, monitor_interval=1.0, sample_duration=0.07,
                            high_thresh=85, critical_thresh=95, quality_levels=None):
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
        self.current_load = 0
        self.last_change_time = 0
        self.lock = threading.Lock()
        self.running = False
        self.load_history = []
        self.max_history_size = 3
        self.min_change_interval = 1.5
        self.monitor_thread = None
        self.last_activity_time = 0
        self._native_cpu_percent = self._init_cpu_percent_func()

    def _init_cpu_percent_func(self):
        system = platform.system()
        if system == "Linux":
            logger.debug("CPU Monitor: Using Linux native method (/proc/stat)")
            return self._cpu_percent_linux
        elif system == "Windows":
            import ctypes
            self._ctypes = ctypes
            logger.debug("CPU Monitor: Using Windows native method (GetSystemTimes)")
            return self._cpu_percent_windows
        else:
            logger.warning(f"CPU Monitor: Unsupported OS {system}, monitoring disabled. Quality will remain MAX.")
            return lambda interval: 0

    def _cpu_percent_linux(self, interval: float) -> int:
        try:
            with open('/proc/stat', 'r') as f:
                fields = f.readline().strip().split()
            if len(fields) < 5:
                return 0
            
            total1 = sum(int(x) for x in fields[1:])
            idle1 = int(fields[4])
            
            time.sleep(interval)
            
            with open('/proc/stat', 'r') as f:
                fields = f.readline().strip().split()
            if len(fields) < 5:
                return 0
                
            total2 = sum(int(x) for x in fields[1:])
            idle2 = int(fields[4])
            
            total_diff = total2 - total1
            idle_diff = idle2 - idle1
            
            if total_diff == 0:
                return 0
                
            return int((1.0 - (idle_diff / total_diff)) * 100.0)
            
        except Exception as e:
            if self.running:
                logger.debug(f"Linux CPU read error: {e}")
            return 0

    def _cpu_percent_windows(self, interval: float) -> int:
        try:
            GetSystemTimes = self._ctypes.windll.kernel32.GetSystemTimes
            
            class FILETIME(self._ctypes.Structure):
                _fields_ = [("dwLowDateTime", self._ctypes.c_ulong),
                            ("dwHighDateTime", self._ctypes.c_ulong)]

            idle1 = FILETIME()
            kernel1 = FILETIME()
            user1 = FILETIME()
            
            if not GetSystemTimes(self._ctypes.byref(idle1), 
                                  self._ctypes.byref(kernel1), 
                                  self._ctypes.byref(user1)):
                return 0
                
            time.sleep(interval)
            
            idle2 = FILETIME()
            kernel2 = FILETIME()
            user2 = FILETIME()
            
            if not GetSystemTimes(self._ctypes.byref(idle2), 
                                  self._ctypes.byref(kernel2), 
                                  self._ctypes.byref(user2)):
                return 0
                
            def filetime_to_int(ft):
                return (ft.dwHighDateTime << 32) + ft.dwLowDateTime
                
            idle1_int = filetime_to_int(idle1)
            idle2_int = filetime_to_int(idle2)
            kernel1_int = filetime_to_int(kernel1)
            kernel2_int = filetime_to_int(kernel2)
            user1_int = filetime_to_int(user1)
            user2_int = filetime_to_int(user2)
            idle_diff = idle2_int - idle1_int
            kernel_diff = kernel2_int - kernel1_int
            user_diff = user2_int - user1_int
            
            total_diff = kernel_diff + user_diff
            
            if total_diff == 0:
                return 0
                
            return int((1.0 - (idle_diff / total_diff)) * 100.0)
            
        except Exception as e:
            if self.running:
                logger.debug(f"Windows CPU read error: {e}")
            return 0

    def start(self):
        with self.lock:
            if self.running:
                return
            self.last_activity_time = time.time()
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.debug("  CPU monitoring ON")

    def stop(self):
        with self.lock:
            if not self.running:
                return
            self.running = False
            logger.debug("  CPU monitoring OFF")

    def record_activity(self):
        self.last_activity_time = time.time()
        if not self.running:
            self.start()

    def _check_idle_and_stop(self):
        if time.time() - self.last_activity_time >= self.idle_timeout:
            self.stop()
            return True
        return False

    def _get_cpu_load(self) -> int:
        try:
            return self._native_cpu_percent(self.sample_duration)
        except Exception as e:
            logger.debug(f"CPU percent error: {e}")
            return 0

    def _add_to_history(self, value: int):
        self.load_history.append(value)
        if len(self.load_history) > self.max_history_size:
            self.load_history.pop(0)

    def _get_average_load(self) -> int:
        if not self.load_history:
            return 0
        return sum(self.load_history) // len(self.load_history)

    def _calculate_target_quality(self, avg_load: int) -> int:
        if self.critical_thresh <= self.high_thresh:
            return 0 if avg_load >= self.high_thresh else self.max_level
            
        if avg_load >= self.critical_thresh:
            return 0
        if avg_load >= self.high_thresh:
            load_ratio = (avg_load - self.high_thresh) / (self.critical_thresh - self.high_thresh)
            return max(0, self.max_level - int(load_ratio * self.max_level))
        return self.max_level

    def _monitor_loop(self):
        while self.running:
            try:
                if self._check_idle_and_stop():
                    break
                    
                cpu_load = self._get_cpu_load()
                
                with self.lock:
                    self._add_to_history(cpu_load)
                    avg_load = self._get_average_load()
                    self.current_load = avg_load
                    
                    target_level = self._calculate_target_quality(avg_load)
                    now = time.time()
                    
                    if target_level != self.current_quality_level:
                        if now - self.last_change_time >= self.min_change_interval:
                            old_level = self.current_quality_level
                            
                            if target_level < self.current_quality_level:
                                if avg_load >= self.critical_thresh:
                                    self.current_quality_level = target_level
                                elif avg_load >= self.high_thresh + 5:
                                    self.current_quality_level = max(target_level, self.current_quality_level - 2)
                                else:
                                    self.current_quality_level -= 1
                            else:
                                self.current_quality_level += 1
                            
                            self.current_quality_level = max(0, min(self.current_quality_level, self.max_level))
                            self.last_change_time = now
                            
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(5)

    def get_current_quality_config(self) -> dict:
        with self.lock:
            return self.quality_levels[self.current_quality_level].copy()

    def get_cpu_load(self) -> int:
        with self.lock:
            return self.current_load

    def get_status(self) -> str:
        with self.lock:
            if not self.running:
                return "CPU Monitor: OFF"
            return f"  CPU Load: {self.current_load}%"