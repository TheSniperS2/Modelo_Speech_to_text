from fastapi import FastAPI, UploadFile, File
import torch
import torchaudio
import os

app = FastAPI()

# Define las clases en el mismo orden que en el entrenamiento
CLASS_NAMES = ['si', 'no']

# Carga el modelo
from train import SimpleNN  # Importa la clase del modelo
model = SimpleNN()
model.load_state_dict(torch.load("model/model.pt"))
model.eval()

def preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.mean(dim=0)
    waveform = waveform[:16000]
    if waveform.shape[0] < 16000:
        waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[0]))
    waveform = waveform / waveform.abs().max()
    return waveform.unsqueeze(0)

@app.post("/predict/")
async def predict(audio: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    audio_path = f"temp/{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    input_tensor = preprocess_audio(audio_path)
    with torch.no_grad():
        prediction = model(input_tensor)
    os.remove(audio_path)
    predicted_class = CLASS_NAMES[prediction.argmax().item()]
    return {"prediction": predicted_class}

# ...existing code...
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print('Model training complete and saved to', model_save_path)
# ...existing code...