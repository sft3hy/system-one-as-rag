"""Resource accounting: what a run actually costs on this machine.

Measured, not assumed. The omlx release notes quote chat TTFT gains from
batched prefill (#4361); a decision request does one prefill and no decode,
so none of those numbers transfer automatically. Everything here is sampled
against the live server process.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any


def _pgrep_rss(pattern: str = "omlx-server") -> tuple[list[tuple[int, int]], float]:
    """(pid, bytes) per match, plus total CPU%."""
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,pcpu=,rss=,comm="],
            capture_output=True, text=True, timeout=5.0, check=False,
        ).stdout
    except Exception:
        return [], 0.0
    rows: list[tuple[int, int]] = []
    cpu = 0.0
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, pcputxt, rss, comm = parts
        if pattern in comm or pattern in line:
            try:
                rows.append((int(pid), int(rss) * 1024))
                cpu += float(pcputxt)
            except ValueError:
                continue
    return rows, cpu


def memory_pressure() -> dict[str, float]:
    """Pages free/speculative/compressed -> GB, from vm_stat."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=5.0, check=False).stdout
    except Exception:
        return {}
    page = 4096.0
    vals: dict[str, float] = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().rstrip(".")
        try:
            n = float(v)
        except ValueError:
            continue
        key = k.strip().lower().replace(" ", "_")
        vals[key] = n * page / 1e9
    out_gb: dict[str, float] = {}
    for key in ("pages_free", "pages_speculative", "pages_inactive",
                "pages_purgeable", "compressed"):
        if key in vals:
            out_gb[key.replace("pages_", "") + "_gb"] = vals[key]
    return out_gb


@dataclass
class ResourceSample:
    t_s: float
    server_rss_gb: float
    server_cpu_pct: float
    free_gb: float
    compressed_gb: float


@dataclass
class ResourceProfile:
    """Peak/resident footprint of one strategy run."""

    strategy: str
    samples: list[ResourceSample] = field(default_factory=list)
    peak_server_rss_gb: float = 0.0
    mean_server_cpu_pct: float = 0.0
    min_free_gb: float = float("inf")
    peak_compressed_gb: float = 0.0
    model_load_s: float | None = None
    cold_first_latency_ms: float | None = None
    resident_after_s: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "peak_server_rss_gb": round(self.peak_server_rss_gb, 2),
            "mean_server_cpu_pct": round(self.mean_server_cpu_pct, 1),
            "min_free_gb": None if self.min_free_gb == float("inf") else round(self.min_free_gb, 2),
            "peak_compressed_gb": round(self.peak_compressed_gb, 2),
            "model_load_s": self.model_load_s,
            "cold_first_latency_ms": self.cold_first_latency_ms,
            "resident_after_s": self.resident_after_s,
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "n_samples": len(self.samples),
        }


class ResourceMonitor:
    """Background sampler. 100 ms is fine-grained enough to catch a model load
    spike and cheap enough to leave running for the whole benchmark."""

    def __init__(self, strategy: str, interval_s: float = 0.1,
                 pattern: str = "omlx-server"):
        self.strategy = strategy
        self.interval = interval_s
        self.pattern = pattern
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.profile = ResourceProfile(strategy=strategy)
        self._t0 = time.perf_counter()

    def start(self) -> ResourceMonitor:
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            rows, cpu = _pgrep_rss(self.pattern)
            if not rows:
                continue
            rss_gb = sum(r for _, r in rows) / 1e9
            mem = memory_pressure()
            s = ResourceSample(
                t_s=time.perf_counter() - self._t0,
                server_rss_gb=rss_gb,
                server_cpu_pct=cpu,
                free_gb=mem.get("free_gb", 0.0) + mem.get("speculative_gb", 0.0),
                compressed_gb=mem.get("compressed_gb", 0.0),
            )
            self.profile.samples.append(s)
            p = self.profile
            p.peak_server_rss_gb = max(p.peak_server_rss_gb, rss_gb)
            p.min_free_gb = min(p.min_free_gb, s.free_gb) if s.free_gb else p.min_free_gb
            p.peak_compressed_gb = max(p.peak_compressed_gb, s.compressed_gb)
            n = len(self.profile.samples)
            p.mean_server_cpu_pct = (
                (p.mean_server_cpu_pct * (n - 1) + cpu) / n
            )

    def stop(self) -> ResourceProfile:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.profile


def admin_request(base_url: str, api_key: str | None, method: str, path: str,
                  body: dict | None = None, timeout: float = 60.0):
    """Raw call against omlx's admin API (discovered via /openapi.json)."""
    import httpx

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    origin = base_url.rsplit("/v1", 1)[0]
    with httpx.Client(base_url=origin, headers=headers, timeout=timeout) as c:
        return c.request(method.upper(), path, json=body)


def reload_models(base_url: str, api_key: str | None):
    """Rescan model dirs so a freshly downloaded checkpoint is discovered."""
    return admin_request(base_url, api_key, "post", "/admin/api/reload", {})


def unload_model(base_url: str, api_key: str | None, model: str) -> bool:
    """Evict a model so a cold run is genuinely cold.

    Real routes (verified against the live server's /openapi.json):
      POST /admin/api/models/{id}/unload
      POST /v1/models/{id}/unload
    """
    for path in (f"/admin/api/models/{model}/unload", f"/v1/models/{model}/unload"):
        try:
            r = admin_request(base_url, api_key, "post", path, {})
            if r.status_code < 400:
                return True
        except Exception:
            continue
    return False


def load_model(base_url: str, api_key: str | None, model: str) -> bool:
    """Warm a model explicitly, so cold-start and warm-start runs are honest
    about which is which."""
    for path in (f"/admin/api/models/{model}/load", f"/v1/models/{model}/load"):
        try:
            r = admin_request(base_url, api_key, "post", path, {}, timeout=600.0)
            if r.status_code < 400:
                return True
        except Exception:
            continue
    return False


def gpu_snapshot() -> dict[str, Any]:
    """Metal residency probe. Not available as a stable counter on macOS;
    the RSS proxy above is the number we report, and this records the attempt
    so the report never implies GPU memory was measured when it was not."""
    return {"gpu_memory_measured": False,
            "note": "macOS exposes no per-process Metal counter; server RSS + "
                   "vm_stat pressure is the reported footprint proxy"}
