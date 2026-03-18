"""GPU device detection — supports NVIDIA (CUDA), Intel Arc (XPU), and CPU fallback."""

from __future__ import annotations

import logging
import os
import subprocess
from functools import lru_cache

logger = logging.getLogger(__name__)


def _import_torch():
    """Lazy torch import to avoid hard failure when torch isn't installed."""
    try:
        import torch
        return torch
    except ImportError:
        return None


@lru_cache(maxsize=1)
def detect_device() -> str:
    """Detect the best available device: 'cuda', 'xpu', or 'cpu'."""
    torch = _import_torch()
    if torch is None:
        logger.warning("PyTorch not installed — defaulting to CPU")
        return "cpu"

    if torch.cuda.is_available():
        logger.info("CUDA device detected: %s", torch.cuda.get_device_name(0))
        return "cuda"

    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401

        if torch is not None and hasattr(torch, "xpu") and torch.xpu.is_available():
            logger.info("Intel XPU device detected: %s", torch.xpu.get_device_name(0))
            return "xpu"
    except ImportError:
        pass

    logger.warning("No GPU detected — falling back to CPU. Training will be very slow.")
    return "cpu"


def get_device() -> str:
    """Return the torch device string."""
    return detect_device()


def get_device_map() -> str:
    """Return the device_map for model loading."""
    device = detect_device()
    if device == "cuda":
        return "auto"
    if device == "xpu":
        return "xpu"
    return "cpu"


def get_dtype(prefer_bf16: bool = True):
    """Return the recommended compute dtype for the detected device."""
    torch = _import_torch()
    if torch is None:
        return None
    device = detect_device()
    if device == "xpu":
        return torch.bfloat16  # XPU works best with bf16
    if device == "cuda":
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def is_cuda() -> bool:
    return detect_device() == "cuda"


def is_xpu() -> bool:
    return detect_device() == "xpu"


def is_cpu() -> bool:
    return detect_device() == "cpu"


def has_bitsandbytes() -> bool:
    """Check if bitsandbytes is available (CUDA-only)."""
    try:
        import bitsandbytes  # noqa: F401
        return is_cuda()
    except ImportError:
        return False


def has_ipex_quantization() -> bool:
    """Check if Intel Extension for Transformers quantization is available."""
    try:
        from intel_extension_for_transformers.transformers import AutoModelForCausalLM as IEFTModel  # noqa: F401
        return is_xpu()
    except ImportError:
        return False


def gpu_status() -> dict:
    """Return GPU status info for the API, supporting NVIDIA and Intel."""
    device = detect_device()
    result: dict = {"device_type": device, "gpus": []}

    if device == "cuda":
        try:
            import GPUtil
            for g in GPUtil.getGPUs():
                result["gpus"].append({
                    "id": g.id,
                    "name": g.name,
                    "memory_total": g.memoryTotal,
                    "memory_used": g.memoryUsed,
                    "memory_free": g.memoryFree,
                    "gpu_util": g.load * 100,
                    "temperature": g.temperature,
                })
        except Exception:
            result["error"] = "CUDA detected but GPUtil unavailable"

    elif device == "xpu":
        torch = _import_torch()
        try:
            count = torch.xpu.device_count()
            for i in range(count):
                props = torch.xpu.get_device_properties(i)
                mem_total = props.total_memory // (1024 * 1024)  # bytes → MB
                # XPU memory tracking
                try:
                    mem_used = torch.xpu.memory_allocated(i) // (1024 * 1024)
                    mem_free = mem_total - mem_used
                except Exception:
                    mem_used = 0
                    mem_free = mem_total
                result["gpus"].append({
                    "id": i,
                    "name": props.name,
                    "memory_total": mem_total,
                    "memory_used": mem_used,
                    "memory_free": mem_free,
                    "gpu_util": None,  # XPU doesn't expose util% easily
                    "temperature": None,
                })
        except Exception:
            # Fallback: try xpu-smi
            try:
                out = subprocess.check_output(["xpu-smi", "discovery"], text=True, timeout=5)
                result["xpu_smi"] = out.strip()
            except Exception:
                pass
            # Fallback: check /dev/dri
            if os.path.exists("/dev/dri"):
                render_nodes = [f for f in os.listdir("/dev/dri") if f.startswith("renderD")]
                if render_nodes:
                    result["gpus"].append({
                        "id": 0,
                        "name": "Intel GPU (detected via /dev/dri)",
                        "memory_total": None,
                        "memory_used": None,
                        "memory_free": None,
                        "gpu_util": None,
                        "temperature": None,
                    })

    else:
        result["error"] = "No GPU detected — running on CPU"

    return result
