# JARVIS - AI Personal Assistant

An advanced AI personal assistant built in Python that responds to voice commands and hand gestures. JARVIS runs entirely on your local hardware (GPU-accelerated) with internet access for self-learning capabilities.

## Features

- **Voice Control** — Speak commands using wake word "JARVIS" with offline speech recognition (Vosk) and online fallback (Google)
- **Hand Gesture Recognition** — Control JARVIS with hand gestures via webcam using MediaPipe (open palm, fist, thumbs up/down, peace sign, pointing)
- **Local AI Brain** — GPU-accelerated LLM inference using llama-cpp-python for fully offline AI processing (supports any GGUF model)
- **Overlay UI** — Transparent, always-on-top chat interface built with PyQt5 that stays out of your way
- **System Control** — Open/close apps, take screenshots, control volume, manage files, type text, lock screen
- **Coding Assistant** — Write, execute, and debug code in Python, JavaScript, Bash, PowerShell, and HTML
- **Web Search & Learning** — Search the internet via DuckDuckGo, fetch web pages, and self-learn new skills
- **Persistent Memory** — SQLite-based memory system stores conversations, learned knowledge, command patterns, and user preferences
- **Self-Learning** — Automatically searches the web to learn skills it doesn't know, caches knowledge for future use
- **Cross-Platform** — Works on Windows and Linux with platform-specific optimizations

## Architecture

```
JARVIS/
├── main.py                    # Entry point (GUI + terminal modes)
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── jarvis/
│   ├── core/
│   │   ├── engine.py          # Main orchestrator connecting all subsystems
│   │   ├── brain.py           # LLM integration (llama-cpp-python)
│   │   ├── memory.py          # SQLite persistent memory & learning
│   │   └── config.py          # YAML configuration manager
│   ├── voice/
│   │   ├── listener.py        # Speech recognition (Vosk offline + Google)
│   │   └── speaker.py         # Text-to-speech (pyttsx3)
│   ├── vision/
│   │   └── gesture.py         # Hand gesture recognition (MediaPipe)
│   ├── actions/
│   │   ├── system.py          # System control (apps, files, OS commands)
│   │   ├── coding.py          # Code execution & formatting
│   │   └── web.py             # Web search & content fetching
│   ├── skills/
│   │   └── skill_manager.py   # Self-learning skill management
│   ├── ui/
│   │   └── overlay.py         # PyQt5 transparent overlay window
│   └── utils/
│       ├── gpu.py             # GPU detection & management
│       └── logger.py          # Logging system
├── data/
│   ├── models/                # LLM and Vosk models (downloaded separately)
│   ├── memory/                # SQLite database
│   └── skills/                # Cached web skills
└── logs/                      # Application logs
```

## Requirements

- **Python** 3.10+
- **GPU** (recommended): NVIDIA (CUDA), AMD (ROCm/OpenCL), or Intel (oneAPI) GPU for accelerated LLM inference
- **Microphone**: For voice commands
- **Webcam**: For hand gesture recognition (optional)
- **OS**: Windows 10/11 or Linux (Ubuntu 20.04+)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/zk-labs-code/JARVIS.git
cd JARVIS
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
# Basic installation
pip install -r requirements.txt

# For GPU support, install llama-cpp-python with the backend matching your GPU:

# NVIDIA (CUDA)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# AMD (ROCm) — requires ROCm toolkit installed
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# AMD / Intel / Universal (OpenCL via CLBlast)
CMAKE_ARGS="-DGGML_CLBLAST=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Intel (oneAPI / SYCL)
CMAKE_ARGS="-DGGML_SYCL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Vulkan (universal — works on most GPUs)
CMAKE_ARGS="-DGGML_VULKAN=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# On Windows, use `set` instead of inline env:
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 4. Download AI Models

#### LLM Model (Required for full AI capabilities)

Download any GGUF-format model and place it in `data/models/llm/`. Recommended models:

```bash
# Create model directory
mkdir -p data/models/llm

# Example: Download a small, fast model (Phi-2 Q4)
# Visit https://huggingface.co/TheBloke and download a GGUF model
# Place the .gguf file in data/models/llm/
```

> **Note**: JARVIS works without an LLM model in "fallback mode" using rule-based command parsing. Download a model for full conversational AI capabilities.

#### Vosk Speech Model (Required for offline voice recognition)

```bash
# Create model directory and download
mkdir -p data/models
cd data/models

# Download the small English model (~40MB)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
cd ../..
```

## Usage

### GUI Mode (Default)

```bash
python main.py
```

Launches JARVIS with a transparent overlay window. Use the text input or voice commands.

### Terminal Mode

```bash
python main.py --no-gui
```

Run JARVIS in the terminal without the overlay UI.

### Command-Line Options

```
python main.py [options]

Options:
  --config, -c PATH    Path to config file (default: config.yaml)
  --no-gui             Run in terminal-only mode
  --no-voice           Disable voice recognition
  --no-gesture         Disable gesture recognition
  --debug              Enable debug logging
```

## Voice Commands

Say **"JARVIS"** followed by your command:

| Command | Example |
|---------|---------|
| Open apps | "JARVIS, open notepad" |
| Close apps | "JARVIS, close chrome" |
| Web search | "JARVIS, search for Python tutorials" |
| Run code | "JARVIS, write a hello world in Python" |
| Screenshot | "JARVIS, take a screenshot" |
| Time/Date | "JARVIS, what time is it?" |
| Volume | "JARVIS, volume up" |
| Learn | "JARVIS, learn about machine learning" |
| File ops | "JARVIS, list files in documents" |
| System info | "JARVIS, system info" |

## Hand Gestures

| Gesture | Action |
|---------|--------|
| Open Palm (5 fingers) | Activate listening mode |
| Closed Fist | Stop listening |
| Thumbs Up | Confirm action |
| Thumbs Down | Cancel action |
| Peace Sign (2 fingers) | Take screenshot |
| Point Up | Scroll up |
| Point Down | Scroll down |

## Configuration

Edit `config.yaml` to customize JARVIS:

- **Voice settings**: Wake word, recognition engine, speech rate
- **Gesture settings**: Camera, detection confidence, gesture-action mappings
- **AI settings**: Model path, context length, temperature, GPU layers
- **UI settings**: Window size, position, opacity, colors, theme
- **System settings**: Allowed operations, restricted paths
- **Web settings**: Search engine, timeout, learning preferences

## GPU Setup

JARVIS automatically detects and uses your GPU (NVIDIA, AMD, or Intel).
The detection pipeline tries multiple methods in order:
PyTorch → nvidia-smi / rocm-smi → GPUtil → OpenCL → Vulkan → lspci / WMI.

### NVIDIA (CUDA)

1. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
2. Install llama-cpp-python with CUDA (see Installation step 3)
3. JARVIS auto-detects your GPU and offloads LLM layers

### AMD (ROCm / OpenCL)

1. Install [ROCm](https://rocm.docs.amd.com/) **or** your distro's OpenCL driver
   - Ubuntu: `sudo apt install rocm-dkms` (ROCm) or `sudo apt install mesa-opencl-icd` (OpenCL)
   - For Fire Pro / Radeon Pro cards, ROCm or the AMDGPU-PRO OpenCL driver is recommended
2. Install llama-cpp-python with the matching backend:
   - ROCm: `CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir`
   - OpenCL: `CMAKE_ARGS="-DGGML_CLBLAST=on" pip install llama-cpp-python --force-reinstall --no-cache-dir`
3. Verify detection: `rocm-smi` or `clinfo` should list your GPU

### Intel (oneAPI / SYCL)

1. Install [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)
2. Install llama-cpp-python with SYCL: `CMAKE_ARGS="-DGGML_SYCL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir`

### Configuration

Edit `config.yaml` under the `gpu:` section:

```yaml
gpu:
  prefer_gpu: true
  memory_fraction: 0.8
  backend: "auto"   # auto, cuda, rocm, sycl, opencl, vulkan, cpu
```

Set `backend` to `"auto"` (default) to let JARVIS pick the best backend, or force a specific one.

## How Self-Learning Works

1. When JARVIS receives a command it doesn't understand, it searches the web
2. Relevant information is fetched, parsed, and stored in the local SQLite database
3. Future similar queries use cached knowledge first (faster, offline)
4. Command patterns are tracked to improve response accuracy over time
5. All learning happens locally — no data leaves your machine

## License

MIT License - See [LICENSE](LICENSE) for details.
