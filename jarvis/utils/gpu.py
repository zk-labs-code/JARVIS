"""GPU management and monitoring for JARVIS.

Supports multiple GPU vendors:
- NVIDIA (CUDA / nvidia-smi)
- AMD (ROCm / rocm-smi / OpenCL via pyopencl)
- Intel (oneAPI / sycl)
- Fallback via GPUtil, Vulkan, lspci, WMI

Automatically detects the best available GPU and configures
the environment for llama-cpp-python acceleration.
"""

import logging
import os
import platform
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger("jarvis.gpu")

# GPU vendor constants
VENDOR_NVIDIA = "nvidia"
VENDOR_AMD = "amd"
VENDOR_INTEL = "intel"
VENDOR_UNKNOWN = "unknown"


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    id: int
    name: str
    vendor: str  # "nvidia", "amd", "intel", "unknown"
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    gpu_load_percent: float
    temperature: float | None
    extra: dict[str, str] = field(default_factory=dict)


class GPUManager:
    """Manages GPU detection, selection, and monitoring.

    Supports NVIDIA (CUDA), AMD (ROCm/OpenCL), and Intel GPUs.
    Auto-detects the best available GPU and sets up environment
    variables for llama-cpp-python and other GPU-accelerated libraries.
    """

    def __init__(self, prefer_gpu: bool = True, memory_fraction: float = 0.8):
        """Initialize GPU manager.

        Args:
            prefer_gpu: Whether to prefer GPU over CPU.
            memory_fraction: Maximum fraction of GPU memory to use.
        """
        self.prefer_gpu = prefer_gpu
        self.memory_fraction = memory_fraction
        self._gpu_available = False
        self._gpu_info: list[GPUInfo] = []
        self._detected_vendor: str = VENDOR_UNKNOWN
        self._cuda_available = False
        self._rocm_available = False
        self._opencl_available = False
        self._vulkan_available = False
        self._detect_all_gpus()

    # ------------------------------------------------------------------ #
    #  Detection pipeline                                                 #
    # ------------------------------------------------------------------ #

    def _detect_all_gpus(self) -> None:
        """Run the full GPU detection pipeline."""
        # 1. PyTorch (supports CUDA and ROCm)
        if self._detect_via_pytorch():
            return

        # 2. Vendor-specific CLI tools
        if self._detect_nvidia_smi():
            return
        if self._detect_rocm_smi():
            return

        # 3. GPUtil (NVIDIA only)
        if self._detect_via_gputil():
            return

        # 4. OpenCL (AMD, Intel, NVIDIA)
        if self._detect_via_opencl():
            return

        # 5. Vulkan (universal)
        if self._detect_via_vulkan():
            return

        # 6. OS-level fallback (lspci / WMI)
        if self._detect_via_system():
            return

        logger.warning("No GPU detected. JARVIS will run on CPU (slower performance).")

    # -- PyTorch (CUDA + ROCm) ------------------------------------------

    def _detect_via_pytorch(self) -> bool:
        """Detect GPUs via PyTorch (supports both CUDA and ROCm)."""
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    mem_total = props.total_mem / (1024**2)
                    vendor = self._vendor_from_name(name)
                    self._gpu_info.append(GPUInfo(
                        id=i, name=name, vendor=vendor,
                        memory_total_mb=mem_total, memory_used_mb=0,
                        memory_free_mb=mem_total, gpu_load_percent=0,
                        temperature=None,
                    ))

                self._gpu_available = True
                self._detected_vendor = (
                    self._gpu_info[0].vendor if self._gpu_info else VENDOR_UNKNOWN
                )
                if self._detected_vendor == VENDOR_AMD:
                    self._rocm_available = True
                    logger.info(f"ROCm available via PyTorch: {gpu_count} AMD GPU(s)")
                else:
                    self._cuda_available = True
                    logger.info(f"CUDA available via PyTorch: {gpu_count} GPU(s)")
                return True

            # PyTorch ROCm build where torch.cuda reports False
            if hasattr(torch, "hip") or "rocm" in torch.__version__.lower():
                logger.info("PyTorch ROCm build detected (HIP backend)")
                self._rocm_available = True
        except ImportError:
            pass
        return False

    # -- NVIDIA via nvidia-smi -------------------------------------------

    def _detect_nvidia_smi(self) -> bool:
        """Detect NVIDIA GPUs via nvidia-smi CLI."""
        if platform.system() == "Darwin":
            return False
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                    "temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return False

            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    temp = (
                        float(parts[5])
                        if len(parts) >= 6 and parts[5]
                        else None
                    )
                    self._gpu_info.append(GPUInfo(
                        id=int(parts[0]), name=parts[1], vendor=VENDOR_NVIDIA,
                        memory_total_mb=float(parts[2]),
                        memory_used_mb=float(parts[3]),
                        memory_free_mb=float(parts[4]),
                        gpu_load_percent=0, temperature=temp,
                    ))

            if self._gpu_info:
                self._gpu_available = True
                self._cuda_available = True
                self._detected_vendor = VENDOR_NVIDIA
                logger.info(
                    f"{len(self._gpu_info)} NVIDIA GPU(s) detected via nvidia-smi"
                )
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    # -- AMD via rocm-smi / rocminfo -------------------------------------

    def _detect_rocm_smi(self) -> bool:
        """Detect AMD GPUs via rocm-smi or rocminfo CLI."""
        if self._detect_rocm_smi_csv():
            return True
        if self._detect_rocm_smi_basic():
            return True
        return self._detect_rocminfo()

    def _detect_rocm_smi_csv(self) -> bool:
        """Detect AMD GPUs using rocm-smi CSV output."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showid", "--showproductname",
                 "--showmeminfo", "vram", "--showtemp", "--csv"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return False
            return self._parse_rocm_csv(result.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _parse_rocm_csv(self, output: str) -> bool:
        """Parse rocm-smi CSV output into GPUInfo entries."""
        lines = output.strip().split("\n")
        if len(lines) < 2:
            return False

        for i, line in enumerate(lines[1:]):
            parts = [p.strip() for p in line.split(",")]
            if not parts:
                continue

            name = "AMD GPU"
            mem_total = 0.0
            mem_used = 0.0
            for p in parts:
                pl = p.lower()
                if any(kw in pl for kw in ["radeon", "firepro", "fire pro", "instinct"]):
                    name = p
                elif pl.endswith(("mb", "mib")):
                    try:
                        val = float(pl.replace("mb", "").replace("mib", "").strip())
                        if mem_total == 0:
                            mem_total = val
                        else:
                            mem_used = val
                    except ValueError:
                        pass

            self._gpu_info.append(GPUInfo(
                id=i, name=name, vendor=VENDOR_AMD,
                memory_total_mb=mem_total, memory_used_mb=mem_used,
                memory_free_mb=mem_total - mem_used,
                gpu_load_percent=0, temperature=None,
            ))

        if self._gpu_info:
            self._gpu_available = True
            self._rocm_available = True
            self._detected_vendor = VENDOR_AMD
            logger.info(f"{len(self._gpu_info)} AMD GPU(s) detected via rocm-smi")
            return True
        return False

    def _detect_rocm_smi_basic(self) -> bool:
        """Detect AMD GPUs using basic rocm-smi output."""
        try:
            result = subprocess.run(
                ["rocm-smi"], capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0 or "GPU" not in result.stdout:
                return False

            gpu_count = sum(
                1 for line in result.stdout.split("\n")
                if "GPU[" in line or (
                    "gpu" in line.lower() and any(c.isdigit() for c in line)
                )
            )
            if gpu_count > 0:
                for i in range(gpu_count):
                    self._gpu_info.append(GPUInfo(
                        id=i, name=f"AMD GPU {i}", vendor=VENDOR_AMD,
                        memory_total_mb=0, memory_used_mb=0, memory_free_mb=0,
                        gpu_load_percent=0, temperature=None,
                    ))
                self._gpu_available = True
                self._rocm_available = True
                self._detected_vendor = VENDOR_AMD
                logger.info(f"{gpu_count} AMD GPU(s) detected via rocm-smi (basic)")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    def _detect_rocminfo(self) -> bool:
        """Detect AMD GPUs using rocminfo command."""
        try:
            result = subprocess.run(
                ["rocminfo"], capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return False

            gpu_id = 0
            in_gpu_agent = False
            current_name = ""

            for line in result.stdout.split("\n"):
                stripped = line.strip()
                if "Agent " in stripped and "GPU" in stripped.upper():
                    in_gpu_agent = True
                    current_name = f"AMD GPU {gpu_id}"
                elif in_gpu_agent and "Name:" in stripped:
                    current_name = stripped.split(":", 1)[1].strip()
                elif in_gpu_agent and ("Pool Size:" in stripped or stripped == ""):
                    if current_name:
                        self._gpu_info.append(GPUInfo(
                            id=gpu_id, name=current_name, vendor=VENDOR_AMD,
                            memory_total_mb=0, memory_used_mb=0, memory_free_mb=0,
                            gpu_load_percent=0, temperature=None,
                        ))
                        gpu_id += 1
                    in_gpu_agent = False

            if self._gpu_info:
                self._gpu_available = True
                self._rocm_available = True
                self._detected_vendor = VENDOR_AMD
                logger.info(f"{len(self._gpu_info)} AMD GPU(s) detected via rocminfo")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    # -- GPUtil (NVIDIA) -------------------------------------------------

    def _detect_via_gputil(self) -> bool:
        """Detect NVIDIA GPUs via GPUtil library."""
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if gpus:
                for gpu in gpus:
                    self._gpu_info.append(GPUInfo(
                        id=gpu.id, name=gpu.name, vendor=VENDOR_NVIDIA,
                        memory_total_mb=gpu.memoryTotal,
                        memory_used_mb=gpu.memoryUsed,
                        memory_free_mb=gpu.memoryFree,
                        gpu_load_percent=gpu.load * 100,
                        temperature=gpu.temperature,
                    ))
                self._gpu_available = True
                self._cuda_available = True
                self._detected_vendor = VENDOR_NVIDIA
                logger.info(f"{len(gpus)} NVIDIA GPU(s) detected via GPUtil")
                return True
        except ImportError:
            pass
        return False

    # -- OpenCL (AMD / Intel / NVIDIA) -----------------------------------

    def _detect_via_opencl(self) -> bool:
        """Detect GPUs via OpenCL (works for AMD, Intel, and NVIDIA)."""
        try:
            import pyopencl as cl

            platforms = cl.get_platforms()
            gpu_id = 0

            for plat in platforms:
                try:
                    devices = plat.get_devices(device_type=cl.device_type.GPU)
                except cl.LogicError:
                    continue

                for dev in devices:
                    name = dev.name.strip()
                    vendor = self._vendor_from_name(f"{plat.vendor} {name}")
                    mem_total = dev.global_mem_size / (1024**2)
                    self._gpu_info.append(GPUInfo(
                        id=gpu_id, name=name, vendor=vendor,
                        memory_total_mb=mem_total, memory_used_mb=0,
                        memory_free_mb=mem_total, gpu_load_percent=0,
                        temperature=None,
                        extra={"platform": plat.name, "opencl_version": dev.version},
                    ))
                    gpu_id += 1

            if self._gpu_info:
                self._gpu_available = True
                self._opencl_available = True
                self._detected_vendor = self._gpu_info[0].vendor
                logger.info(
                    f"{len(self._gpu_info)} GPU(s) detected via OpenCL "
                    f"(vendor: {self._detected_vendor})"
                )
                return True
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"OpenCL detection failed: {e}")
        return False

    # -- Vulkan ----------------------------------------------------------

    def _detect_via_vulkan(self) -> bool:
        """Detect GPUs via vulkaninfo CLI."""
        try:
            result = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return False

            gpu_id = 0
            for line in result.stdout.split("\n"):
                if "deviceName" in line:
                    sep = "=" if "=" in line else ":"
                    name = line.split(sep)[-1].strip()
                    vendor = self._vendor_from_name(name)
                    self._gpu_info.append(GPUInfo(
                        id=gpu_id, name=name, vendor=vendor,
                        memory_total_mb=0, memory_used_mb=0, memory_free_mb=0,
                        gpu_load_percent=0, temperature=None,
                        extra={"api": "vulkan"},
                    ))
                    gpu_id += 1

            if self._gpu_info:
                self._gpu_available = True
                self._vulkan_available = True
                self._detected_vendor = self._gpu_info[0].vendor
                logger.info(
                    f"{len(self._gpu_info)} GPU(s) detected via Vulkan "
                    f"(vendor: {self._detected_vendor})"
                )
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    # -- System-level fallback -------------------------------------------

    def _detect_via_system(self) -> bool:
        """Detect GPUs via OS-level tools (lspci on Linux, WMI on Windows)."""
        system = platform.system()
        if system == "Linux":
            return self._detect_lspci()
        elif system == "Windows":
            return self._detect_wmi()
        return False

    def _detect_lspci(self) -> bool:
        """Detect GPUs via lspci on Linux."""
        try:
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return False

            gpu_id = 0
            for line in result.stdout.split("\n"):
                ll = line.lower()
                if "vga" in ll or "3d controller" in ll or "display" in ll:
                    name = line.split(": ", 1)[-1] if ": " in line else line
                    vendor = self._vendor_from_name(name)
                    self._gpu_info.append(GPUInfo(
                        id=gpu_id, name=name.strip(), vendor=vendor,
                        memory_total_mb=0, memory_used_mb=0, memory_free_mb=0,
                        gpu_load_percent=0, temperature=None,
                        extra={"detection": "lspci"},
                    ))
                    gpu_id += 1

            if self._gpu_info:
                self._gpu_available = True
                self._detected_vendor = self._gpu_info[0].vendor
                logger.info(
                    f"{len(self._gpu_info)} GPU(s) detected via lspci "
                    f"(vendor: {self._detected_vendor})"
                )
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    def _detect_wmi(self) -> bool:
        """Detect GPUs via WMI on Windows."""
        try:
            result = subprocess.run(
                [
                    "powershell", "-Command",
                    "Get-WmiObject Win32_VideoController | "
                    "Select-Object Name, AdapterRAM | Format-List",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return False

            gpu_id = 0
            name = ""

            for line in result.stdout.split("\n"):
                line = line.strip()
                if line.startswith("Name"):
                    name = line.split(":", 1)[-1].strip()
                elif line.startswith("AdapterRAM") and name:
                    try:
                        mem = float(line.split(":", 1)[-1].strip()) / (1024**2)
                    except (ValueError, IndexError):
                        mem = 0.0
                    vendor = self._vendor_from_name(name)
                    self._gpu_info.append(GPUInfo(
                        id=gpu_id, name=name, vendor=vendor,
                        memory_total_mb=mem, memory_used_mb=0,
                        memory_free_mb=mem, gpu_load_percent=0,
                        temperature=None, extra={"detection": "wmi"},
                    ))
                    gpu_id += 1
                    name = ""

            if self._gpu_info:
                self._gpu_available = True
                self._detected_vendor = self._gpu_info[0].vendor
                logger.info(
                    f"{len(self._gpu_info)} GPU(s) detected via WMI "
                    f"(vendor: {self._detected_vendor})"
                )
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _vendor_from_name(name: str) -> str:
        """Determine GPU vendor from device/platform name string."""
        n = name.lower()
        if any(kw in n for kw in [
            "nvidia", "geforce", "quadro", "tesla", "rtx", "gtx",
        ]):
            return VENDOR_NVIDIA
        if any(kw in n for kw in [
            "amd", "radeon", "firepro", "fire pro", "instinct",
            "advanced micro", "ati ", "navi", "vega", "polaris", "ellesmere",
        ]):
            return VENDOR_AMD
        if any(kw in n for kw in [
            "intel", "arc ", "iris", "uhd graphics", "hd graphics",
        ]):
            return VENDOR_INTEL
        return VENDOR_UNKNOWN

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def is_available(self) -> bool:
        """Check if any GPU is available and enabled."""
        return self._gpu_available and self.prefer_gpu

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA (NVIDIA) is available."""
        return self._cuda_available

    @property
    def rocm_available(self) -> bool:
        """Check if ROCm (AMD) is available."""
        return self._rocm_available

    @property
    def opencl_available(self) -> bool:
        """Check if OpenCL is available."""
        return self._opencl_available

    @property
    def detected_vendor(self) -> str:
        """Get the detected GPU vendor."""
        return self._detected_vendor

    @property
    def gpus(self) -> list[GPUInfo]:
        """Get list of available GPUs."""
        return self._gpu_info

    def get_best_gpu(self) -> GPUInfo | None:
        """Get the GPU with the most free memory."""
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

        # If memory info is unavailable, attempt full offload
        if best_gpu.memory_free_mb == 0 and best_gpu.memory_total_mb == 0:
            logger.info("GPU memory info unavailable; attempting full offload")
            return -1

        available_mem = best_gpu.memory_free_mb * self.memory_fraction
        if available_mem >= model_size_mb:
            return -1  # Offload all layers

        # Estimate partial offloading
        ratio = available_mem / model_size_mb
        estimated_layers = int(ratio * 40)  # Assume ~40 layers for typical models
        return max(estimated_layers, 1)

    def get_llama_cpp_backend(self) -> str:
        """Determine the best llama-cpp-python backend for the detected GPU.

        Returns:
            Backend string: cuda, rocm, sycl, opencl, vulkan, or cpu.
        """
        if self._cuda_available or self._detected_vendor == VENDOR_NVIDIA:
            return "cuda"
        if self._rocm_available or self._detected_vendor == VENDOR_AMD:
            return "rocm"
        if self._detected_vendor == VENDOR_INTEL:
            return "sycl"
        if self._opencl_available:
            return "opencl"
        if self._vulkan_available:
            return "vulkan"
        return "cpu"

    def get_install_instructions(self) -> str:
        """Get GPU-specific installation instructions for llama-cpp-python."""
        backend = self.get_llama_cpp_backend()
        instructions = {
            "cuda": (
                "# NVIDIA CUDA backend\n"
                "CMAKE_ARGS=\"-DGGML_CUDA=on\" "
                "pip install llama-cpp-python --force-reinstall --no-cache-dir"
            ),
            "rocm": (
                "# AMD ROCm backend (requires ROCm toolkit)\n"
                "CMAKE_ARGS=\"-DGGML_HIPBLAS=on\" "
                "pip install llama-cpp-python --force-reinstall --no-cache-dir\n"
                "# Alternative - CLBlast (OpenCL):\n"
                "CMAKE_ARGS=\"-DGGML_CLBLAST=on\" "
                "pip install llama-cpp-python --force-reinstall --no-cache-dir"
            ),
            "sycl": (
                "# Intel oneAPI/SYCL backend\n"
                "CMAKE_ARGS=\"-DGGML_SYCL=on\" "
                "pip install llama-cpp-python --force-reinstall --no-cache-dir"
            ),
            "opencl": (
                "# OpenCL backend (universal)\n"
                "CMAKE_ARGS=\"-DGGML_CLBLAST=on\" "
                "pip install llama-cpp-python --force-reinstall --no-cache-dir"
            ),
            "vulkan": (
                "# Vulkan backend (universal)\n"
                "CMAKE_ARGS=\"-DGGML_VULKAN=on\" "
                "pip install llama-cpp-python --force-reinstall --no-cache-dir"
            ),
            "cpu": (
                "# CPU only (no GPU acceleration)\n"
                "pip install llama-cpp-python"
            ),
        }
        return instructions.get(backend, instructions["cpu"])

    def setup_environment(self) -> None:
        """Set up environment variables for GPU usage based on detected vendor."""
        if not self.is_available:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            logger.info("GPU disabled, using CPU")
            return

        best_gpu = self.get_best_gpu()
        if best_gpu is None:
            return

        if best_gpu.vendor == VENDOR_NVIDIA:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu.id)
            logger.info(
                f"Set CUDA_VISIBLE_DEVICES={best_gpu.id} ({best_gpu.name})"
            )
        elif best_gpu.vendor == VENDOR_AMD:
            # ROCm uses HIP_VISIBLE_DEVICES
            os.environ["HIP_VISIBLE_DEVICES"] = str(best_gpu.id)
            os.environ["GPU_DEVICE_ORDINAL"] = str(best_gpu.id)
            logger.info(
                f"Set HIP_VISIBLE_DEVICES={best_gpu.id}, "
                f"GPU_DEVICE_ORDINAL={best_gpu.id} ({best_gpu.name})"
            )
        elif best_gpu.vendor == VENDOR_INTEL:
            os.environ["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{best_gpu.id}"
            logger.info(
                f"Set ONEAPI_DEVICE_SELECTOR=level_zero:{best_gpu.id} "
                f"({best_gpu.name})"
            )
        else:
            logger.info(
                f"GPU {best_gpu.id} selected: {best_gpu.name} "
                f"(vendor: {best_gpu.vendor})"
            )

    def get_status_summary(self) -> str:
        """Get a human-readable GPU status summary."""
        if not self._gpu_info:
            return "No GPU detected - running on CPU"

        backend = self.get_llama_cpp_backend()
        lines = [f"GPU Backend: {backend.upper()}"]

        for gpu in self._gpu_info:
            if gpu.memory_total_mb > 0:
                mem_pct = (
                    (gpu.memory_used_mb / gpu.memory_total_mb * 100)
                    if gpu.memory_total_mb else 0
                )
                line = (
                    f"GPU {gpu.id}: {gpu.name} [{gpu.vendor.upper()}] | "
                    f"Memory: {gpu.memory_used_mb:.0f}/"
                    f"{gpu.memory_total_mb:.0f} MB ({mem_pct:.1f}%)"
                )
            else:
                line = f"GPU {gpu.id}: {gpu.name} [{gpu.vendor.upper()}]"

            if gpu.temperature is not None:
                line += f" | Temp: {gpu.temperature}C"
            lines.append(line)

        return "\n".join(lines)
