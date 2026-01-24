
import time
import psutil
import torch
import os
import threading
from collections import defaultdict

class ResourceTracker:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.stats = defaultdict(lambda: {"count": 0, "time": 0.0, "cpu_time": 0.0, "max_rss": 0, "read_bytes": 0, "write_bytes": 0, "gpu_max_mem": 0})
        self.current_stage = None
        self.stage_start_time = 0
        self.stage_start_cpu = 0
        self.stage_start_io = None
        
        self.gpu_handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assume GPU 0 for now
        except Exception as e:
            print(f"Warning: pynvml not initialized: {e}")

    def start(self, stage_name):
        if self.current_stage is not None:
             self.stop()
        
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        self.stage_start_cpu = self.process.cpu_times().user + self.process.cpu_times().system
        
        try:
            io = self.process.io_counters()
            self.stage_start_io = (io.read_chars, io.write_chars) # read_chars/write_chars are linux specific but usually available
        except AttributeError:
             self.stage_start_io = (0, 0)
        
        # Reset per-stage peaks if needed, but for now we track max over calls
        pass

    def stop(self):
        if self.current_stage is None:
            return

        end_time = time.time()
        end_cpu = self.process.cpu_times().user + self.process.cpu_times().system
        
        read_bytes_delta = 0
        write_bytes_delta = 0
        try:
            io = self.process.io_counters()
            current_io = (io.read_chars, io.write_chars)
            if self.stage_start_io:
                read_bytes_delta = current_io[0] - self.stage_start_io[0]
                write_bytes_delta = current_io[1] - self.stage_start_io[1]
        except AttributeError:
            pass

        duration = end_time - self.stage_start_time
        cpu_duration = end_cpu - self.stage_start_cpu
        
        rss = self.process.memory_info().rss
        
        gpu_mem = 0
        if self.gpu_handle:
            try:
                import pynvml
                info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_mem = info.used
            except:
                pass
        elif torch.cuda.is_available():
             gpu_mem = torch.cuda.memory_allocated()

        s = self.stats[self.current_stage]
        s["count"] += 1
        s["time"] += duration
        s["cpu_time"] += cpu_duration
        s["max_rss"] = max(s["max_rss"], rss)
        s["read_bytes"] += read_bytes_delta
        s["write_bytes"] += write_bytes_delta
        s["gpu_max_mem"] = max(s["gpu_max_mem"], gpu_mem)

        self.current_stage = None

    def track(self, stage_name):
        return ResourceContext(self, stage_name)

    def print_stats(self):
        print("\n" + "="*80)
        print(f"{'Stage':<15} | {'Count':<5} | {'Time(s)':<10} | {'CPU(s)':<8} | {'MaxRSS(MB)':<10} | {'Read(MB)':<8} | {'Write(MB)':<8} | {'GPU(MB)':<8}")
        print("-" * 80)
        for stage, data in self.stats.items():
            print(f"{stage:<15} | {data['count']:<5} | {data['time']:<10.4f} | {data['cpu_time']:<8.4f} | {data['max_rss']/1024/1024:<10.2f} | {data['read_bytes']/1024/1024:<8.2f} | {data['write_bytes']/1024/1024:<8.2f} | {data['gpu_max_mem']/1024/1024:<8.2f}")
        print("="*80 + "\n")

    def save_stats(self, path):
        with open(path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"{'Stage':<20} | {'Count':<5} | {'Time(s)':<10} | {'CPU(s)':<8} | {'MaxRSS(MB)':<10} | {'Read(MB)':<8} | {'Write(MB)':<8} | {'GPU(MB)':<8}\n")
            f.write("-" * 80 + "\n")
            for stage, data in self.stats.items():
                f.write(f"{stage:<20} | {data['count']:<5} | {data['time']:<10.4f} | {data['cpu_time']:<8.4f} | {data['max_rss']/1024/1024:<10.2f} | {data['read_bytes']/1024/1024:<8.2f} | {data['write_bytes']/1024/1024:<8.2f} | {data['gpu_max_mem']/1024/1024:<8.2f}\n")
            f.write("="*80 + "\n")


class ResourceContext:
    def __init__(self, tracker, stage_name):
        self.tracker = tracker
        self.stage_name = stage_name

    def __enter__(self):
        self.tracker.start(self.stage_name)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.stop()
