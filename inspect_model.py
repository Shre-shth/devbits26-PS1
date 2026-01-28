import onnxruntime as ort

model_path = "silero_vad.onnx"
try:
    session = ort.InferenceSession(model_path)
    print("--- Inputs ---")
    for i in session.get_inputs():
        print(f"Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
    print("\n--- Outputs ---")
    for o in session.get_outputs():
        print(f"Name: {o.name}, Shape: {o.shape}, Type: {o.type}")
except Exception as e:
    print(f"Error inspecting model: {e}")
