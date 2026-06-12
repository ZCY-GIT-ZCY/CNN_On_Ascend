"""
环境监控模块 - 在 benchmark 运行期间持续记录 NPU 状态、系统负载、温度等。

用法：
  from env_monitor import EnvMonitor

  monitor = EnvMonitor(log_dir="logs")
  monitor.start(interval=2.0)  # 每2秒采样一次

  # ... 运行 benchmark ...

  monitor.stop()
  monitor.report()  # 打印统计摘要

也可以独立运行：
  python env_monitor.py --duration 30 --interval 1
"""
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class EnvMonitor:
    """后台线程持续监控 NPU 和系统状态。"""

    def __init__(self, log_dir: str = "logs", label: str = ""):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.label = label.replace("/", "_").replace(" ", "_")
        self._samples: List[Dict] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

    def _sample_npu(self) -> Dict:
        """采集 NPU 状态。"""
        info = {"timestamp": time.time(), "datetime": datetime.now().isoformat()}

        # npu-smi 总体状态
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "health", "-i", "0"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if "Health" in line:
                    info["npu_health"] = line.split(":")[-1].strip()
        except Exception:
            info["npu_health"] = "unknown"

        # npu-smi 基本信息
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-i", "0"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if "Temperature" in line:
                    info["npu_temp_c"] = float(line.split(":")[-1].strip())
                if "Power(W)" in line:
                    parts = line.split()
                    for p in parts:
                        try:
                            val = float(p)
                            if val < 100:  # 功率值
                                info["npu_power_w"] = val
                        except ValueError:
                            pass
        except Exception:
            pass

        # 查询更多温度信息
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "temp", "-i", "0"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if "Temperature" in line:
                    info["npu_temp_c"] = float(line.split(":")[-1].strip())
        except Exception:
            pass

        # 查询使用率
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "usages", "-i", "0"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if "Memory Usage" in line or "Memory-Usage" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if "/" in p and "MB" in p:
                            mem_parts = p.split("/")
                            try:
                                info["npu_mem_used_mb"] = float(mem_parts[0])
                                info["npu_mem_total_mb"] = float(mem_parts[1].replace("MB", "").strip())
                            except ValueError:
                                pass
        except Exception:
            pass

        return info

    def _sample_system(self) -> Dict:
        """采集系统级状态。"""
        info = {}
        # CPU 负载
        try:
            with open("/proc/loadavg") as f:
                parts = f.read().strip().split()
                info["load_1m"] = float(parts[0])
                info["load_5m"] = float(parts[1])
                info["load_15m"] = float(parts[2])
        except Exception:
            pass

        # CPU 温度
        for path in [
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/devices/virtual/thermal/thermal_zone0/temp",
        ]:
            try:
                with open(path) as f:
                    info["cpu_temp_c"] = float(f.read().strip()) / 1000.0
                break
            except Exception:
                continue

        # CPU 频率
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
                info["cpu_freq_mhz"] = float(f.read().strip()) / 1000.0
        except Exception:
            pass

        # 内存
        try:
            result = subprocess.run(
                ["free", "-m"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if line.startswith("Mem:"):
                    parts = line.split()
                    info["mem_total_mb"] = float(parts[1])
                    info["mem_used_mb"] = float(parts[2])
                    info["mem_free_mb"] = float(parts[3])
        except Exception:
            pass

        # NPU 频率（如果有）
        for i in range(4):
            try:
                path = f"/sys/class/devfreq/fdab0000.npu/cur_freq"
                with open(path) as f:
                    info["npu_freq_hz"] = int(f.read().strip())
                break
            except Exception:
                try:
                    # 另一种可能的路径
                    import glob
                    paths = glob.glob(f"/sys/class/devfreq/*npu*/cur_freq")
                    if paths:
                        with open(paths[0]) as f:
                            info["npu_freq_hz"] = int(f.read().strip())
                    break
                except Exception:
                    pass

        return info

    def _collect(self):
        """采集一轮样本。"""
        sample = {}
        sample.update(self._sample_system())
        sample.update(self._sample_npu())
        with self._lock:
            self._samples.append(sample)

    def _loop(self, interval: float):
        """后台采样循环。"""
        while not self._stop_event.is_set():
            try:
                self._collect()
            except Exception as e:
                # 不因采样失败影响主流程
                with self._lock:
                    self._samples.append({"error": str(e), "timestamp": time.time()})
            self._stop_event.wait(interval)

    def start(self, interval: float = 2.0):
        """启动后台监控。"""
        if self._thread and self._thread.is_alive():
            return
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(interval,), daemon=True
        )
        self._thread.start()

    def stop(self):
        """停止监控并保存日志。"""
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=10)
            self._thread = None

        self._save()

    def _save(self):
        """保存采样数据到文件。"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_part = f"_{self.label}" if self.label else ""
        filename = self.log_dir / f"env_monitor_{ts}{label_part}.json"
        with self._lock:
            data = {
                "start_time": self._start_time,
                "end_time": time.time(),
                "sample_count": len(self._samples),
                "samples": self._samples,
            }
        filename.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def report(self) -> str:
        """生成环境监控摘要报告。"""
        with self._lock:
            if not self._samples:
                return "No samples collected."

            lines = ["=== 环境监控摘要 ==="]

            # NPU 温度
            temps = [s.get("npu_temp_c") for s in self._samples if "npu_temp_c" in s]
            if temps:
                lines.append(f"NPU 温度: min={min(temps):.1f}°C, "
                             f"max={max(temps):.1f}°C, "
                             f"avg={sum(temps)/len(temps):.1f}°C")

            # NPU 频率
            freqs = [s.get("npu_freq_hz") for s in self._samples if "npu_freq_hz" in s]
            if freqs:
                lines.append(f"NPU 频率: {freqs[0]/1e6:.0f} MHz (consistent={len(set(freqs))==1})")

            # CPU 温度
            cpu_temps = [s.get("cpu_temp_c") for s in self._samples if "cpu_temp_c" in s]
            if cpu_temps:
                lines.append(f"CPU 温度: min={min(cpu_temps):.1f}°C, "
                             f"max={max(cpu_temps):.1f}°C")

            # 系统负载
            loads = [s.get("load_1m") for s in self._samples if "load_1m" in s]
            if loads:
                lines.append(f"系统负载 (1m): min={min(loads):.2f}, "
                             f"max={max(loads):.2f}, "
                             f"avg={sum(loads)/len(loads):.2f}")

            # NPU Health
            healths = [s.get("npu_health") for s in self._samples if "npu_health" in s]
            if healths:
                health_set = set(healths)
                lines.append(f"NPU Health: {', '.join(health_set)}")

            return "\n".join(lines)

    def get_summary_stats(self) -> Dict:
        """返回关键统计信息的字典。"""
        with self._lock:
            stats = {"sample_count": len(self._samples)}
            for key in ["npu_temp_c", "cpu_temp_c", "load_1m", "npu_freq_hz",
                        "npu_power_w", "npu_mem_used_mb"]:
                vals = [s.get(key) for s in self._samples if key in s]
                if vals:
                    stats[key] = {
                        "min": float(min(vals)),
                        "max": float(max(vals)),
                        "mean": float(sum(vals) / len(vals)),
                        "stable": len(set(round(v, 1) for v in vals)) == 1,
                    }
            return stats


def monitor_cli():
    """独立运行：python env_monitor.py --duration 30 --interval 1"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=30, help="监控时长（秒）")
    parser.add_argument("--interval", type=float, default=1.0, help="采样间隔（秒）")
    args = parser.parse_args()

    monitor = EnvMonitor()
    monitor.start(interval=args.interval)
    print(f"Monitoring for {args.duration}s... (interval={args.interval}s)")
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
    monitor.stop()
    print(monitor.report())


if __name__ == "__main__":
    monitor_cli()
