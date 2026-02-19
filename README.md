# Real-Time AI Voice Agent 🤖🎙️

**Team Name:** HindKeSItarey

## 📖 Project Overview

A low-latency, real-time voice AI agent designed to hold natural conversations. This project uniquely integrates state-of-the-art speech recognition, a powerful Large Language Model (LLM) for intelligence, and fast text-to-speech synthesis into a cohesive system. It connects to telephony infrastructure via **Asterisk**, allowing you to call the bot or have it call you.

## 🏗️ Architecture

The system follows a pipeline architecture ensuring minimal latency between user speech and agent response.

```mermaid
graph TD
    User([User Phone]) <-->|RTP / SIP| Asterisk[Asterisk PBX]
    Asterisk <-->|WebSocket / RTP| Bot[Voice Bot (Python)]
    
    subgraph "Voice Bot Internal Pipeline"
        Bot -->|Audio Stream| VAD[Silero VAD]
        VAD -->|Speech Segment| STT[Faster-Whisper]
        STT -->|Text| Brain[LLM (Google Gemini)]
        Brain -->|Text Stream| TTS[Piper TTS]
        TTS -->|Audio Stream| Bot
    end
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Telephony** | **Asterisk 16+** | SIP Server & VoIP Gateway. Handles calls. |
| **Interface** | **ARI (Asterisk REST Interface)** | Controls calls and media flow. |
| **Speech-to-Text** | **Faster-Whisper** | Transcribes user speech audio to text efficiently. |
| **Intelligence** | **Google Gemini Flash** | Generates intelligent, context-aware responses. |
| **Text-to-Speech** | **Piper TTS** | Converts text response back to high-quality audio. |
| **Voice Activity** | **Silero VAD** | Detects human speech to handle turn-taking and barge-ins. |

---

## 🚀 Setup Instructions

Follow these steps to set up the project on a fresh Linux machine (Debian/Ubuntu recommended).

### 1. System Prerequisites

Install system-level dependencies for audio processing and Python building.

```bash
sudo apt-get update
sudo apt-get install -y python3-dev portaudio19-dev build-essential ffmpeg git
```

### 2. Asterisk Installation & Configuration (Required)

You **MUST** have an Asterisk server running to use this bot.

1.  **Install Asterisk** (if not already installed):
    Follow [official Asterisk installation guides](https://wiki.asterisk.org/wiki/display/AST/Installing+Asterisk+From+Source) for your specific OS.

2.  **Configure `ari.conf`**:
    Enable the ARI interface and create a user.
    *File:* `/etc/asterisk/ari.conf`
    ```ini
    [general]
    enabled = yes
    pretty = yes
    allowed_origins = *

    [brain]
    type = user
    read_only = no
    password = 1234
    password_format = plain
    ```

3.  **Configure `extensions.conf`**:
    Route calls to the Stasis application named `ai-bot`.
    *File:* `/etc/asterisk/extensions.conf`
    ```ini
    [default]
    exten => 1000,1,NoOp(Route to AI Bot)
     same => n,Stasis(ai-bot)
     same => n,Hangup()
    ```
    *Note: `1000` is the extension you will dial to reach the bot.*

4.  **Restart Asterisk**:
    ```bash
    sudo systemctl restart asterisk
    ```

### 3. Clone & Setup Project

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Shre-shth/devbits26-PS1.git
    cd devbits26-PS1
    ```

2.  **Create Virtual Environment**:
    recommended to isolate dependencies.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### 4. Application Configuration

1.  **Environment Variables**:
    Create a `.env` file in the project root to store sensitive keys.
    ```bash
    cp .env.example .env  # If .env.example exists, otherwise create .env
    ```
    **Add your Google Gemini API Key:**
    *File:* `.env`
    ```ini
    GOOGLE_API_KEY=your_actual_google_api_key_here
    # Optional overrides
    # LOG_LEVEL=INFO
    # CALL_TYPE=inbound
    ```

### 5. Download AI Models

The project requires specific models for VAD and TTS to be present locally.

1.  **Silero VAD Model**:
    ```bash
    wget -O silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
    ```

2.  **Piper TTS Voice (Amy Low)**:
    ```bash
    wget -O en_US-amy-low.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx?download=true"
    wget -O en_US-amy-low.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx.json?download=true"
    ```

---

## ▶️ Running the Bot

Ensure Asterisk is running before starting the bot.

### Option A: Using the Helper Script (Recommended)
This script handles library paths for NVIDIA/CUDA if present and activates the environment.

```bash
chmod +x run.sh
./run.sh
```

### Option B: Manual Execution
```bash
# Ensure .venv is active
source .venv/bin/activate

# Add NVIDIA libs to path if using GPU (optional but recommended for speed)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib

python main.py
```

### Expected Output
When successful, you will see logs indicating connection to Asterisk:
- `[ARI] Connected to ARI.`
- `Listening on 127.0.0.1:4000...` (Ready for RTP audio)

---

## 📞 How to Use

1.  **Inbound Call**: Configure a softphone (like Zoiper or MicroSIP) to register with your Asterisk server. Dial extension `1000`. The bot should answer and greet you.
2.  **Outbound Call**: You can trigger the bot to call you (requires configuration in `trigger_outbound.py` or similar logic).

---

## ❓ Troubleshooting

| Issue | Possible Cause | Solution |
| :--- | :--- | :--- |
| **ModuleNotFoundError** | Dependencies not installed | Run `pip install -r requirements.txt` in your venv. |
| **[ARI] Connection Refused** | Asterisk down or wrong creds | Check `sudo asterisk -rvvv`, verify `ari.conf` password matches `src/config.py` (or default `1234`). |
| **No Audio** | RTP port block / NAT | Ensure firewall allows UDP ports 10000-20000 (RTP) and port 4000 (Local RTP). Check if Asterisk and Bot are on same network. |
| **"Bad magic number"** | Asterisk Codec mismatch | Ensure Asterisk is configured for `ulaw` or `alaw` (PCMU/PCMA). |

---

**Developed with ❤️ by Team HindKeSItarey**
