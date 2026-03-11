# SMART SECRETARY (MODULE B) - TECHNICAL DOCUMENTATION

## 1. Overview
The Smart Secretary is a high-performance audio transcription and meeting summarization tool designed to process call recordings. It utilizes state-of-the-art Speech-to-Text (STT) and Large Language Models (LLM) to convert raw audio into timestamped transcripts and actionable Minutes of Meeting (MOM).

## 2. Core Technology: Faster-Whisper
We have chosen **Faster-Whisper** as the primary transcription engine. Faster-Whisper is a reimplementation of OpenAI's Whisper model using CTranslate2, a fast inference engine for Transformer models.

### Why Faster-Whisper?
*   **Speed**: Up to 4x faster than the original OpenAI Whisper implementation while maintaining the same accuracy.
*   **Efficiency**: Consumes significantly less memory (VRAM/RAM) due to advanced quantization techniques (e.g., int8, float16).
*   **Stability**: Built on C++, it offers a more stable environment for long-running batch processing of audio files.

## 3. The Power of the `large-v3` Model
For this implementation, we have defaulted to the **`large-v3`** model, the most advanced version of Whisper available.

### Excellence in Indian Languages
Indian languages (Hindi, Bengali, Marathi, Telugu, etc.) are linguistically complex with varied accents and dialects. 
*   **Massive Dataset**: `large-v3` was trained on an even larger dataset of multilingual audio compared to its predecessors, leading to a significantly lower Word Error Rate (WER) for Indian languages.
*   **Better Contextual Understanding**: It excels at handling "Hinglish" (code-switching between Hindi and English) and other mixed-language conversations common in the Indian corporate and social environment.
*   **Robustness**: It is highly resistant to background noise and low-quality recordings, which is crucial for real-world call recordings.

## 4. Hardware and GPU Requirements
To run the `large-v3` model effectively, the following hardware is recommended:

### GPU (Recommended for High Speed)
*   **Minimum VRAM**: 6GB (using `int8_float16` quantization).
*   **Recommended VRAM**: 8GB+ for smooth processing.
*   **Architecture**: NVIDIA GPU with CUDA support (Compute Capability 7.0+ recommended).
*   **Optimization**: Our implementation uses `compute_type="int8_float16"` which reduces VRAM usage by nearly 50% without a noticeable drop in accuracy, allowing the large model to run on consumer-grade GPUs.

### CPU (Fallback)
*   The script will automatically fallback to CPU if no GPU is detected.
*   **Requirement**: Multicore CPU (AMD/Intel) with at least 16GB of System RAM.
*   **Note**: CPU transcription is significantly slower (up to 10-20x slower than GPU).

## 5. Implementation Details in Module B
The module is designed for ease of use and historical tracking:

*   **Audio Preprocessing**: Automatically converts any MP3 input to 16kHz Mono PCM (the exact format Whisper requires).
*   **Timestamping**: Generates a `call_transcript.txt` with precise `[start -> end]` time frames for every segment.
*   **MOM Generation**: Integration with Gemini (Brain) to generate professional summaries in `minutes_of_Meeting.txt`.
*   **File Management**:
    *   `call_transcript.txt` / `minutes_of_Meeting.txt`: Contains the results of the LATEST run.
    *   `[filename]_transcript.txt` / `[filename]_mom.txt`: Contains specific historical records named after your audio file.

## 6. How to Run
You can run the secretary from any directory, but the script is located in `module B Smart Secretary`.
```bash 
"./module B Smart Secretary/run_secretary.sh" "path/to/your/audio.mp3"
```
Or if you are already inside the `module B Smart Secretary` folder:
```bash
./run_secretary.sh "path/to/your/audio.mp3"
```
