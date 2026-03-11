# SMART SECRETARY (MODULE B) - TECHNICAL DOCUMENTATION

## 1. Overview (Sarvam AI)
The Smart Secretary is a high-performance audio transcription and meeting summarization tool designed to process call recordings. It utilizes state-of-the-art Speech-to-Text (STT) and Large Language Models (LLM) to convert raw audio into timestamped transcripts and actionable Minutes of Meeting (MOM).

## 2. Core Technology: High-Speed Parallel Processing
We use **Sarvam AI's standard Speech-to-Text API** with a custom high-performance parallel chunking engine. This approach is designed to provide the fastest possible transcription for call recordings of any length.

### Why Parallel Chunking?
*   **Extreme Speed**: Instead of waiting in a slow batch queue (which can take minutes for even short files), the audio is locally sliced into 28-second "micro-chunks" and transcribed concurrently in the cloud.
*   **Real-time Feel**: A 10-minute recording can be transcribed in roughly 30-60 seconds—a **10x speed improvement** over standard batch processing.
*   **Reliability**: Smaller API requests are more stable and less prone to timeout than single large uploads.
*   **Seamless Stitching**: The module automatically reassembles the transcribed chunks into a coherent full transcript for the AI brain.

### Technical Implementation:
1.  **FFmpeg Splicing**: Audio is instantly split into 28-second segments without re-encoding, preserving quality and speed.
2.  **Multithreading**: We utilize a `ThreadPoolExecutor` with 8 concurrent workers to hit the Sarvam AI endpoints simultaneously.
3.  **Strict Ordering**: Each chunk is indexed and mapped, ensuring the final transcript remains chronologically perfect regardless of which chunk finishes first.
4.  **MOM Acceleration**: By accelerating the transcription phase, the overall Minutes of Meeting (MOM) generation starts much sooner, significantly reducing total turnaround time.

## 3. Hardware Requirements
Because transcription is now API-driven, the hardware requirements for Module B are minimal:

*   **Internet Connection**: Required to connect to the Sarvam AI API.
*   **CPU/RAM**: Any modern multicore processor with 4GB+ RAM is sufficient.
*   **GPU**: No longer required for transcription in Module B.

---

## 3. Features
*   **MOM Generation**: Integration with Gemini (Brain) to generate professional summaries in `minutes_of_Meeting.txt`.
*   **Multilingual Support**: Optimized for Indian languages and code-switching (Hinglish).
*   **Unified File Management**:
    *   `call_transcript.txt` / `minutes_of_Meeting.txt`: Contains the results of the LATEST run.
    *   `[filename]_transcript.txt` / `[filename]_mom.txt`: Contains specific historical records named after your audio file.

---

## 4. How to Run
Ensure your **`SARVAM_API_KEY`** and **`GOOGLE_API_KEY`** are correctly set in the `.env` file at the project root.

You can execute the Smart Secretary from the **project root** using any of the following methods:

### Method A: Python Module Execution (Recommended)
Ideal for developers who have already activated the virtual environment:
```bash
# 1. Activate environment
source .venv/bin/activate
# 2. Run as module (assuming you are in devbits26-PS1 directory)
python3 -m "module B Smart Secretary.secretary" "path/to/recording.mp3"
```

### Method B: Direct Python Execution 
Use this if you want to use the virtual environment's Python directly:
```bash
.venv/bin/python3 "module B Smart Secretary/secretary.py" "path/to/recording.mp3"
```

### Method C: The Script (Change the script with correct paths) (Not recommended)
The fastest way to run the secretary with all environment variables pre-configured.
```bash
./"module B Smart Secretary/run_secretary.sh" "path/to/recording.mp3"
```

---

## 5. Output and Results
After execution, the following files are generated in the `module B Smart Secretary/` directory:

| File Type | Naming Convention | Purpose |
| :--- | :--- | :--- |
| **Latest Transcript** | `call_transcript.txt` | The transcript of the most recent run. |
| **Latest MOM** | `minutes_of_Meeting.txt` | The MOM of the most recent run. |
| **Historical Record** | `[filename]_transcript.txt` | Dated transcript for reference. |
| **Historical Record** | `[filename]_mom.txt` | Dated MOM for reference. |
