from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
import time


@dataclass
class EnergyReading:
    wall_seconds: float
    cpu_joules: float | None
    gpu_joules: float | None

    @property
    def total_joules(self) -> float | None:
        vals = [v for v in (self.cpu_joules, self.gpu_joules) if v is not None]
        return sum(vals) if vals else None

    @property
    def total_kwh(self) -> float | None:
        return None if self.total_joules is None else self.total_joules / 3_600_000.0


def _read_rapl() -> float | None:
    values = []
    for path in glob.glob("/sys/class/powercap/intel-rapl*/energy_uj") + glob.glob("/sys/class/powercap/intel-rapl/intel-rapl:*/energy_uj"):
        try:
            values.append(float(Path(path).read_text().strip()) / 1e6)
        except Exception:
            pass
    return sum(values) if values else None


def _read_nvml() -> float | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        total = 0.0
        found = False
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                total += float(pynvml.nvmlDeviceGetTotalEnergyConsumption(h)) / 1000.0
                found = True
            except Exception:
                pass
        return total if found else None
    except Exception:
        return None


class EnergyMeter:
    def __enter__(self):
        self.t0 = time.perf_counter()
        self.cpu0 = _read_rapl()
        self.gpu0 = _read_nvml()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.t1 = time.perf_counter()
        self.cpu1 = _read_rapl()
        self.gpu1 = _read_nvml()
        self.reading = EnergyReading(
            wall_seconds=self.t1 - self.t0,
            cpu_joules=None if self.cpu0 is None or self.cpu1 is None else max(0.0, self.cpu1 - self.cpu0),
            gpu_joules=None if self.gpu0 is None or self.gpu1 is None else max(0.0, self.gpu1 - self.gpu0),
        )
