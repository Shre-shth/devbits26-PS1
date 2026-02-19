# 🌌 Real-Time AI Voice Agent: Project Starfire 🚀

> **"Connecting Intelligence at the Speed of Sound"**

### 👨‍🚀 Mission Control: **HindKeSitarey**

---

## � Mission Brief

Welcome to a next-generation **Low-Latency Voice AI** system. Like a satellite relay, this agent captures your voice, beams it to a powerful LLM core, and synthesizes a human-like response in real-time. It integrates state-of-the-art speech recognition, generative intelligence, and high-fidelity synthesis into a cohesive orbital system.

The agent docks efficiently with **Asterisk PBX**, allowing seamless two-way communication via standard telephony protocols.

---

## 🛰️ System Constellation (Architecture)

Our architecture is designed for speed and stability, ensuring minimal signal decay (latency) across the pipeline.

```mermaid
graph TD
    User([👨‍🚀 User Comms]) <-->|RTP / SIP| Asterisk[📡 Asterisk Gateway]
    Asterisk <-->|WebSocket data stream| Bot[🛸 Voice Bot Core]
    
    subgraph "The AI Nebula"
        Bot -->|Audio Waveform| VAD[🔈 Silero VAD (Sensor)]
        VAD -->|Signal Segment| STT[📝 Faster-Whisper (Decoder)]
        STT -->|Text Data| Brain[🧠 Gemini Flash (Intelligence)]
        Brain -->|Response Stream| TTS[🗣️ Piper TTS (Synthesizer)]
        TTS -->|Audio Waveform| Bot
    end
```

---

## ⚛️ Propulsion Systems (Tech Stack)

Powered by a fusion of high-performance technologies.

| Module | Component | Function |
| :--- | :--- | :--- |
| **Gateway** | **Asterisk 16+** | The docking station for all incoming SIP/VoIP calls. |
| **Protocol** | **ARI (Asterisk REST Interface)** | Command & Control link between the PBX and Python Core. |
| **Decoder** | **Faster-Whisper** | Hyper-fast transcription engine. |
| **Core Intelligence** | **Google Gemini Flash** | The central brain processing context and generating responses. |
| **Synthesizer** | **Piper TTS** | Low-latency text-to-speech engine. |
| **Sensors** | **Silero VAD** | Voice Activity Detection to handle user interruptions. |

---

## 🚀 Launch Sequence (Setup)

Prepare your ground station (Linux Machine) for deployment.

### 1. 🛠️ Ground Support Equipment (Prerequisites)
Install the necessary system-level libraries for audio processing and build tools.

```bash
sudo apt-get update
sudo apt-get install -y python3-dev portaudio19-dev build-essential ffmpeg git
```

### 2. 📡 Telemetry Uplink (Asterisk Setup)
**CRITICAL:** The Asterisk Gateway must be operational.

1.  **Deploy Asterisk**: Install following [official guides](https://wiki.asterisk.org/wiki/display/AST/Installing+Asterisk+From+Source).
2.  **Configure ARI (`/etc/asterisk/ari.conf`)**:
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
3.  **Establish Flight Path (`/etc/asterisk/extensions.conf`)**:
    ```ini
    [default]
    exten => 1000,1,NoOp(Route to AI Core)
     same => n,Stasis(ai-bot)
     same => n,Hangup()
    ```
4.  **Reboot Systems**: `sudo systemctl restart asterisk`

### 3. 💾 Initialize Repository
```bash
git clone https://github.com/Shre-shth/devbits26-PS1.git
cd devbits26-PS1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. 🔑 Encryption Keys (.config)
Create your environmental config.
```bash
cp .env.example .env
```
Populate `.env` with your **Google Gemini API Key**:
```ini
GOOGLE_API_KEY=AIzaSy...
```

### 5. 📦 Download Payload (Models)
Acquire the neural network weights required for onboard processing.

```bash
# VAD Sensor
wget -O silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx

# Voice Synthesis Module (Amy Low)
wget -O en_US-amy-low.onnx "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx?download=true"
wget -O en_US-amy-low.onnx.json "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/low/en_US-amy-low.onnx.json?download=true"
```

---

## 🚦 Ignition (Running the Bot)

Ensure all systems are nominal before starting the main engine.

### Option A: Auto-Launch (Recommended)
Automatically configures the NVIDIA/CUDA trajectory if GPU is detected.

```bash
chmod +x run.sh
./run.sh
```

### Option B: Manual Override
```bash
source .venv/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib
python main.py
```

**Expected Telemetry:**
- `[ARI] Connected to ARI.` ✅
- `Listening on 127.0.0.1:4000...` ✅

---

## 🎧 Communication Protocols (Usage)

1.  **Inbound Hail**: Register your SIP Softphone (Zoiper/MicroSIP) to the Asterisk Server. Dial Coordinates **1000**.
2.  **Outbound Hail**: Trigger the bot to initiate contact (requires `trigger_outbound.py` execution).

---

## ⚠️ Anomaly Detection (Troubleshooting)

| Alert | Diagnosis | Corrective Action |
| :--- | :--- | :--- |
| **ModuleNotFoundError** | Missing Modules | `pip install -r requirements.txt` |
| **Connection Refused** | Gateway Offline | Check `sudo asterisk -rvvv`. Verify `ari.conf` credentials. |
| **No Audio** | Signal Jamming (NAT/Firewall) | Check UDP ports 10000-20000 (RTP) and 4000. |
| **Bad Magic Number** | Codec Mismatch | Ensure Asterisk uses `ulaw` or `alaw`. |

---

**Crafted across the stars by Team HindKeSitarey 🌟**
