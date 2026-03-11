# SMART SECRETARY (MODULE B) - TECHNICAL DOCUMENTATION

## 1. Overview
The Smart Secretary is a high-performance audio transcription and meeting summarization tool designed to process call recordings. It utilizes state-of-the-art Speech-to-Text (STT) and Large Language Models (LLM) to convert raw audio into timestamped transcripts and actionable Minutes of Meeting (MOM).

## 2. Core Technology: Sarvam AI (High-Speed Parallel Processing)
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
*   **Timestamping**: Generates a `call_transcript.txt` with precise `[start -> end]` time frames for every segment.
*   **MOM Generation**: Integration with Gemini (Brain) to generate professional summaries in `minutes_of_Meeting.txt`.
*   **File Management**:
    *   `call_transcript.txt` / `minutes_of_Meeting.txt`: Contains the results of the LATEST run.
    *   `[filename]_transcript.txt` / `[filename]_mom.txt`: Contains specific historical records named after your audio file.

## 6. How to Run
Must ensure that the api key is set in the .env file. SARVAM_API_KEY=your_api_key
You can run the secretary from any directory, but the script is located in `module B Smart Secretary`.
```bash 
"./module B Smart Secretary/run_secretary.sh" "path/to/your/audio.mp3"
```
Or if you are already inside the `module B Smart Secretary` folder:
```bash
./run_secretary.sh "path/to/your/audio.mp3"
```
