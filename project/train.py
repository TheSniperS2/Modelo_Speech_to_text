import os
import torchaudio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(16000, 128)  # Assuming 1 second of audio at 16kHz
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 3)  # Ahora 3 clases

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Custom dataset for loading audio files
class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.audio_files = []
        self.labels = []
        self.class_names = ['si', 'no']
        self.load_data()
        print(f"Encontrados {len(self.audio_files)} audios en {self.data_dir}")

    def load_data(self):
        for label, class_dir in enumerate(self.class_names):
            class_path = os.path.join(self.data_dir, class_dir)
            if not os.path.exists(class_path):
                continue
            for file in os.listdir(class_path):
                if file.endswith('.wav'):
                    self.audio_files.append(os.path.join(class_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # Convert to mono
        waveform = waveform[:16000]  # Trim or pad to 1 second
        if waveform.shape[0] < 16000:
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[0]))  # Pad to 1 second
        return waveform, torch.tensor(self.labels[idx])

def train_model(data_dir, model_save_path, num_epochs=30, batch_size=16, learning_rate=0.001):
    dataset = AudioDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Crear la carpeta del modelo si no existe
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print('Model training complete and saved to', model_save_path)

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data"))
    MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "model/model.pt"))
    train_model(data_dir=DATA_DIR, model_save_path=MODEL_DIR, num_epochs=50)