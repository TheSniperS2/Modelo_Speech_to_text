# Audio Recognition Project

This project is designed for a simple audio recognition system that distinguishes between two classes: "Sí" and "No". It utilizes various Python libraries for audio processing, model training, and inference.

## Project Structure

```
project
├── data
│   ├── si          # Directory for audio files recorded for the "Sí" class
│   └── no          # Directory for audio files recorded for the "No" class
├── model
│   └── model.pt    # Trained model for voice recognition
├── record.py       # Script to record audio and save it based on user input
├── train.py        # Script to train the voice recognition model
├── inference.py     # Script for model inference on new audio inputs
├── requirements.txt # List of dependencies required for the project
└── README.md       # Documentation for the project
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd project
pip install -r requirements.txt
```

## Usage

1. **Recording Audio**: Run `record.py` to start recording audio. Click the "Sí" button to save audio in the `data/si` directory or the "No" button to save audio in the `data/no` directory.
   ```bash
   python record.py
   ```

2. **Training the Model**: After collecting audio samples, run `train.py` to train the voice recognition model using the recorded audio.
   ```bash
   python train.py
   ```

3. **Model Inference**: Use `inference.py` to load the trained model and predict whether a new audio input corresponds to "Sí" or "No".
   ```bash
   python inference.py
   ```

## Dependencies

The project requires the following Python packages:

- PyTorch
- torchaudio
- sounddevice or pyaudio
- FastAPI (optional)
- tkinter (optional)
- numpy
- scipy
- librosa (optional)
- matplotlib (optional)
- scikit-learn

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.