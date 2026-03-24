"""GPU management and monitoring for JARVIS."""

import logging
import os
import platform
from dataclasses import dataclass

logger = logging.getLogger("jarvis.gpu")


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    id: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    gpu_load_percent: float
    temperature: float | None


class GPUManager:
    """Manages GPU detection, selection, and monitoring."""

    def __init__(self, prefer_gpu: bool = True, memory_fraction: float = 0.8):
        """Initialize GPU manager.

        Args:
            prefer_gpu: Whether to prefer GPU over CPU.
            memory_fraction: Maximum fraction of GPU memory to use.
        """
        self.prefer_gpu = prefer_gpu
        self.memory_fraction = memory_fraction
        self._gpu_available = False
        self._cuda_available = False
        self._gpu_info: list[GPUInfo] = []
        self._detect_gpu()

    def _detect_gpu(self) -> None:
        """Detect available GPUs on the system."""
        # Check CUDA availability
        try:
            import torch
            self._cuda_available = torch.cuda.is_available()
            if self._cuda_available:
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    name = torch.cuda.get_device_name(i)
                    mem_total = torch.cuda.get_device_properties(i).total_mem / (1024**2)
                    self._gpu_info.append(GPUInfo(
                        id=i,
                        name=name,
                        memory_total_mb=mem_total,
                        memory_used_mb=0,
                        memory_free_mb=mem_total,
                        gpu_load_percent=0,
                        temperature=None,
                    ))
                self._gpu_available = True
                logger.info(f"CUDA available: {gpu_count} GPU(s) detected via PyTorch")
                return
        except ImportError:
            pass

        # Fallback to GPUtil
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                for gpu in gpus:
                    self._gpu_info.append(GPUInfo(
                        id=gpu.id,
                        name=gpu.name,
                        memory_total_mb=gpu.memoryTotal,
                        memory_used_mb=gpu.memoryUsed,
                        memory_free_mb=gpu.memoryFree,
                        gpu_load_percent=gpu.load * 100,
                        temperature=gpu.temperature,
                    ))
                self._gpu_available = True
                logger.info(f"GPU available: {len(gpus)} GPU(s) detected via GPUtil")
                return
        except ImportError:
            pass

        # Check for NVIDIA GPU via nvidia-smi
        if platform.system() != "Darwin":
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 5:
                            self._gpu_info.append(GPUInfo(
                                id=int(parts[0]),
                                name=parts[1],
                                memory_total_mb=float(parts[2]),
                                memory_used_mb=float(parts[3]),
                                memory_free_mb=float(parts[4]),
                                gpu_load_percent=0,
                                temperature=None,
                            ))
                    self._gpu_available = bool(self._gpu_info)
                    if self._gpu_available:
                        logger.info(
                            f"GPU available: {len(self._gpu_info)} GPU(s) detected via nvidia-smi"
                        )
                        return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        logger.warning("No GPU detected. JARVIS will run on CPU (slower performance).")

    @property
    def is_available(self) -> bool:
        """Check if GPU is available."""
        return self._gpu_available and self.prefer_gpu

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available

    @property
    def gpus(self) -> list[GPUInfo]:
        """Get list of available GPUs."""
        return self._gpu_info

    def get_best_gpu(self) -> GPUInfo | None:
        """Get the GPU with the most free memory.

        Returns:
            Best GPU info or None if no GPU available.
        """
        if not self._gpu_info:
            return None
        return max(self._gpu_info, key=lambda g: g.memory_free_mb)

    def get_gpu_layers(self, model_size_mb: float = 4000) -> int:
        """Calculate optimal number of GPU layers for LLM loading.

        Args:
            model_size_mb: Approximate model size in MB.

        Returns:
            Number of layers to offload to GPU (-1 for all, 0 for CPU only).
        """
        if not self.is_available:
            return 0

        best_gpu = self.get_best_gpu()
        if best_gpu is None:
            return 0

        available_mem = best_gpu.memory_free_mb * self.memory_fraction
        if available_mem >= model_size_mb:
            return -1  # Offload all layers

        # Estimate partial offloading
        ratio = available_mem / model_size_mb
        estimated_layers = int(ratio * 40)  # Assume ~40 layers for typical models
        return max(estimated_layers, 1)

    def setup_environment(self) -> None:
        """Set up environment variables for GPU usage."""
        if self.is_available:
            best_gpu = self.get_best_gpu()
            if best_gpu is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu.id)
                logger.info(f"Set CUDA_VISIBLE_DEVICES={best_gpu.id} ({best_gpu.name})")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("GPU disabled, using CPU")

    def get_status_summary(self) -> str:
        """Get a human-readable GPU status summary.

        Returns:
            Status string.
        """
        if not self._gpu_info:
            return "No GPU detected - running on CPU"

        lines = []
        for gpu in self._gpu_info:
            mem_pct = (gpu.memory_used_mb / gpu.memory_total_mb * 100) if gpu.memory_total_mb else 0
            line = (
                f"GPU {gpu.id}: {gpu.name} | "
                f"Memory: {gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f} MB ({mem_pct:.1f}%)"
            )
            if gpu.temperature is not None:
                line += f" | Temp: {gpu.temperature}C"
            lines.append(line)
        return "\n".join(lines)
