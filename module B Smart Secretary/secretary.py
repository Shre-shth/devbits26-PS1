import os
import sys
import ctypes
import numpy as np

# --- NVIDIA Library Workaround ---
try:
    # 1. Collect potential search paths
    search_paths = []
    if "LD_LIBRARY_PATH" in os.environ:
        search_paths.extend(os.environ["LD_LIBRARY_PATH"].split(":"))
    
    # Add common .venv paths relative to this file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_paths.append(os.path.join(project_root, ".venv/lib/python3.12/site-packages/nvidia/cublas/lib"))
    search_paths.append(os.path.join(project_root, ".venv/lib/python3.12/site-packages/nvidia/cudnn/lib"))

    for path in search_paths:
        if not path or not os.path.exists(path):
            continue
        
        # libcublas requires libcublasLt, so load Lt first if present
        lib_lt = os.path.join(path, "libcublasLt.so.12")
        if os.path.exists(lib_lt):
            try:
                ctypes.CDLL(lib_lt, mode=ctypes.RTLD_GLOBAL)
            except Exception as e:
                pass 

        lib_cublas = os.path.join(path, "libcublas.so.12")
        if os.path.exists(lib_cublas):
            try:
                ctypes.CDLL(lib_cublas, mode=ctypes.RTLD_GLOBAL)
            except Exception as e:
                pass

        # Load all .so.8 files for cuDNN if present
        for filename in os.listdir(path):
            if filename.endswith(".so.8"):
                 try:
                     filepath = os.path.join(path, filename)
                     ctypes.CDLL(filepath, mode=ctypes.RTLD_GLOBAL)
                 except Exception:
                     pass 
except Exception as e:
    print(f"[Warn] NVIDIA library workaround failed: {e}")
# -----------------------------

from faster_whisper import WhisperModel, decode_audio
from scipy.io import wavfile
from src.brain import Brain
from dotenv import load_dotenv

# Add parent directory to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def process_recording(mp3_path, model_size="large-v3"):
    """
    Processes a call recording (mp3), transcribes it, and generates MOM.
    """
    if not os.path.exists(mp3_path):
        print(f"Error: File '{mp3_path}' not found.")
        return

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment.")
        return

    # 1. Load Faster-Whisper Model
    device = "cpu"
    compute_type = "int8"
    
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            print("GPU Detected! Switching to CUDA.")
            device = "cuda"
            # Use int8_float16 to significantly reduce VRAM usage for large models
            compute_type = "int8_float16" if "large" in model_size else "float16"
    except Exception as e:
        print(f"GPU Check Failed: {e}. Defaulting to CPU.")

    print(f"Loading Whisper model: {model_size} on {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print("Whisper model loaded.")

    # 2. Preprocess and Transcribe MP3
    print(f"Preprocessing and Transcribing '{mp3_path}'...")
    
    # Manually decode audio to 16kHz mono (what Whisper hears)
    audio = decode_audio(mp3_path, sampling_rate=16000)
    
    module_b_dir = os.path.dirname(os.path.abspath(__file__))
    base_name = os.path.basename(mp3_path)
    audio_base = os.path.splitext(base_name)[0]

    segments, info = model.transcribe(audio, beam_size=5)
    
    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

    transcript = ""
    formatted_transcript = ""
    for segment in segments:
        line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
        print(line)
        transcript += f"{segment.text} "
        formatted_transcript += f"{line}\n"
    
    transcript = transcript.strip()
    formatted_transcript = formatted_transcript.strip()
    
    if not transcript:
        print("No speech detected in the recording.")
        return

    # 3. Generate MOM
    print("\nGenerating Minutes of Meeting...")
    brain = Brain(api_key=api_key)
    # Give Brain the clean transcript for better analysis
    mom = brain.generate_mom_from_transcript(transcript)
    
    print("\n--- MINUTES OF MEETING ---")
    print(mom)
    print("--------------------------")

    # 4. Save results in module B
    # Fixed filenames (for easy access to the latest run)
    fixed_transcript = os.path.join(module_b_dir, "transcript.txt")
    fixed_mom = os.path.join(module_b_dir, "mom.txt")
    
    # Input-specific filenames (for history)
    specific_transcript = os.path.join(module_b_dir, f"{audio_base}_transcript.txt")
    specific_mom = os.path.join(module_b_dir, f"{audio_base}_mom.txt")

    # Write formatted transcript (with timestamps) to both
    with open(fixed_transcript, "w") as f:
        f.write(formatted_transcript)
    with open(specific_transcript, "w") as f:
        f.write(formatted_transcript)
    
    # Write MOM to both
    with open(fixed_mom, "w") as f:
        f.write(mom)
    with open(specific_mom, "w") as f:
        f.write(mom)

    print(f"\nResults saved in '{module_b_dir}':")
    print(f"- Latest: transcript.txt, mom.txt")
    print(f"- Specific: {audio_base}_transcript.txt, {audio_base}_mom.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m module\\ B.secretary <path_to_mp3_file> [model_size]")
        sys.exit(1)
    
    mp3_file = sys.argv[1]
    size = sys.argv[2] if len(sys.argv) > 2 else "large-v3"
    
    # Simple fix for module path imports if run as script
    # Ensure src is importable
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    process_recording(mp3_file, size)
