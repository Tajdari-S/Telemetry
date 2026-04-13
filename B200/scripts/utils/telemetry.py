"""
Tier-1 NVML Telemetry Collector
Mirrors SPAR-GPU-monitoring collect_nvml_telemetry.py methodology.
Polls pynvml at configurable Hz and returns a pandas DataFrame.
"""
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import pynvml

pynvml.nvmlInit()

@dataclass
class GPUSample:
    ts: float
    gpu_idx: int
    gpu_util: float
    mem_util: float
    mem_used_mb: float
    mem_total_mb: float
    power_w: float
    temp_c: float
    sm_clock_mhz: float
    mem_clock_mhz: float
    gr_engine_active: float   # NvLink Rx bytes (approx active)
    nvlink_rx_mb: float
    nvlink_tx_mb: float
    pcie_rx_mb: float
    pcie_tx_mb: float
    ecc_errors: int

def _safe(fn, default=0.0):
    try:
        return fn()
    except pynvml.NVMLError:
        return default

def sample_gpu(handle, idx: int) -> GPUSample:
    util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
    power  = _safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
    temp   = _safe(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    sm_clk = _safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
    mem_clk= _safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
    ecc    = _safe(lambda: pynvml.nvmlDeviceGetTotalEccErrors(
                       handle, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                       pynvml.NVML_VOLATILE_ECC), 0)

    # NVLink counters (sum across all links)
    nvlink_rx = nvlink_tx = 0.0
    link_count = _safe(lambda: pynvml.NVML_NVLINK_MAX_LINKS, 18)
    for lnk in range(link_count):
        try:
            c = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(handle, lnk, 0)
            nvlink_rx += c['rx'] / 1e6
            nvlink_tx += c['tx'] / 1e6
        except Exception:
            break

    pcie_rx = _safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1e3)  # KB->MB
    pcie_tx = _safe(lambda: pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1e3)

    return GPUSample(
        ts=time.time(), gpu_idx=idx,
        gpu_util=util.gpu, mem_util=util.memory,
        mem_used_mb=mem.used / 1e6, mem_total_mb=mem.total / 1e6,
        power_w=power, temp_c=temp,
        sm_clock_mhz=sm_clk, mem_clock_mhz=mem_clk,
        gr_engine_active=util.gpu,
        nvlink_rx_mb=nvlink_rx, nvlink_tx_mb=nvlink_tx,
        pcie_rx_mb=pcie_rx, pcie_tx_mb=pcie_tx,
        ecc_errors=int(ecc),
    )


class TelemetryRecorder:
    """Background thread that polls all GPUs at `hz` Hz."""
    def __init__(self, gpu_indices: List[int] = None, hz: float = 10.0):
        self.hz = hz
        n = pynvml.nvmlDeviceGetCount()
        self.indices = gpu_indices if gpu_indices is not None else list(range(n))
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.indices]
        self._samples: List[GPUSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> pd.DataFrame:
        self._stop.set()
        self._thread.join()
        return self.to_df()

    def _loop(self):
        interval = 1.0 / self.hz
        while not self._stop.is_set():
            t0 = time.perf_counter()
            for h, idx in zip(self.handles, self.indices):
                self._samples.append(sample_gpu(h, idx))
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, interval - elapsed))

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([s.__dict__ for s in self._samples])
