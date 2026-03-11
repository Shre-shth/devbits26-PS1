import os
import sys
import time
import subprocess
import concurrent.futures
from sarvamai import SarvamAI
from src.brain import Brain
from dotenv import load_dotenv

def split_audio_ffmpeg(audio_path, output_dir, segment_time=28):
    """
    Splits audio into small segments using ffmpeg.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Clear directory if it exists
    for f in os.listdir(output_dir):
        if f.startswith("segment_"):
            os.remove(os.path.join(output_dir, f))

    segment_pattern = os.path.join(output_dir, "segment_%03d.mp3")
    
    # Using -c copy to split without re-encoding (instant)
    # If copy fails because of bitstream issues, we might need to re-encode,
    # but for most mp3/wav it's fine.
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-c", "copy",
        segment_pattern
    ]
    
    print(f"Quick-splitting audio into {segment_time}s chunks...")
    subprocess.run(cmd, check=True, capture_output=True)
    
    segments = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("segment_")])
    return segments

def transcribe_chunk(api_key, chunk_path, model):
    """
    Transcribes a single small chunk (< 30s) via standard Sarvam API.
    A new client is created per thread to be safe.
    """
    try:
        client = SarvamAI(api_subscription_key=api_key)
        with open(chunk_path, "rb") as f:
            response = client.speech_to_text.transcribe(
                file=f,
                model=model,
                mode="transcribe"
            )
        # Extract transcript.
        # Sarvam API returns an object; getattr handles potential string/obj types.
        transcript = getattr(response, 'transcript', str(response))
        return transcript.strip()
    except Exception as e:
        # Don't fail the whole process for one chunk if we can avoid it
        print(f"\n[Warning] Chunk {os.path.basename(chunk_path)} failed: {e}")
        return ""

def process_recording(audio_path, model_size="saaras:v3"):
    """
    High-Speed Parallel Processing:
    Divides audio into <30s slices and transcribes them concurrently.
    """
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        return

    load_dotenv()
    sarvam_api_key = os.getenv("SARVAM_API_KEY")
    if not sarvam_api_key:
        print("Error: SARVAM_API_KEY not found in .env.")
        return

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY not found in environment.")
        return

    file_name = os.path.basename(audio_path)
    audio_base = os.path.splitext(file_name)[0]
    module_b_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create temp directory for chunks
    temp_dir = os.path.join(module_b_dir, f"temp_{audio_base}")
    
    try:
        # 1. Split audio
        chunks = split_audio_ffmpeg(audio_path, temp_dir)
        num_chunks = len(chunks)
        print(f"Created {num_chunks} chunks for parallel processing.")

        # 2. Parallel Transcription
        print(f"Transcribing {num_chunks} chunks in parallel (please wait)...")
        results = []
        # Using 5 workers to be polite to the API/Connection, can increase if needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Future to track index to maintain order
            future_to_idx = {executor.submit(transcribe_chunk, sarvam_api_key, chunk, model_size): i 
                             for i, chunk in enumerate(chunks)}
            
            # Initialize placeholder results list
            results = [""] * num_chunks
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                sys.stdout.write("█")
                sys.stdout.flush()

        print("\nParallel transcription complete.")
        full_transcript = " ".join([r for r in results if r])
        formatted_transcript = full_transcript

    except Exception as e:
        print(f"\nParallel Processing Failed: {e}")
        return
    finally:
        # Cleanup temp chunks
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)

    if not full_transcript.strip():
        print("No speech detected in the recording.")
        return

    # 3. Generate MOM
    print("\nGenerating Minutes of Meeting...")
    brain = Brain(api_key=google_api_key)
    # Give Brain the clean transcript for better analysis
    mom = brain.generate_mom_from_transcript(full_transcript)
    
    print("\n--- MINUTES OF MEETING ---")
    print(mom)
    print("--------------------------")

    # 4. Save results in module B
    # Historical names: call_transcript.txt and minutes_of_Meeting.txt
    fixed_transcript = os.path.join(module_b_dir, "call_transcript.txt")
    fixed_mom = os.path.join(module_b_dir, "minutes_of_Meeting.txt")
    
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
    print(f"- Latest: call_transcript.txt, minutes_of_Meeting.txt")
    print(f"- Specific: {audio_base}_transcript.txt, {audio_base}_mom.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m "module B Smart Secretary.secretary" <path_to_audio_file> [model_name]')
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "saaras:v3"
    
    # Ensure src is importable (needed if running as a script)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    process_recording(audio_file, model)
