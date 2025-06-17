# Audio Recognition Project

Este proyecto implementa un sistema simple de reconocimiento de audio para distinguir entre las palabras "Sí" y "No" usando Python, PyTorch y FastAPI.

## Estructura del Proyecto

```
project
├── data
│   ├── si          # Directorio para archivos de audio grabados para la clase "Sí"
│   └── no          # Directorio para archivos de audio grabados para la clase "No"
├── model
│   └── model.pt    # Modelo entrenado para el reconocimiento de voz
├── record.py       # Interfaz gráfica para grabar audio y probar el modelo
├── train.py        # Script para entrenar el modelo de reconocimiento de voz
├── inference.py    # API FastAPI para la inferencia del modelo en nuevas entradas de audio
├── requirements.txt # Lista de dependencias requeridas para el proyecto
└── README.md       # Documentación del proyecto
```

## Instalación

1. Clona el repositorio y entra a la carpeta `project`:
   ```bash
   git clone <repository-url>
   cd project
   ```

2. Instala las dependencias requeridas:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### 1. Grabar audios de entrenamiento

Ejecuta la interfaz gráfica para grabar audios de "sí" y "no":

```bash
python record.py
```

- Usa el botón **"Grabar 'Sí' (entrenar)"** para guardar audios en `data/si/`.
- Usa el botón **"Grabar 'No' (entrenar)"** para guardar audios en `data/no/`.
- Graba al menos 5-10 ejemplos por clase para mejores resultados.

### 2. Entrenar el modelo

Desde la terminal, ejecuta:

```bash
python train.py
```

Esto entrenará el modelo con los audios guardados y generará el archivo `model/model.pt`.  
Puedes ajustar la cantidad de épocas de entrenamiento modificando el parámetro `num_epochs` en `train.py`.

### 3. Probar el modelo (inferir)

1. Inicia el servidor FastAPI:
   ```bash
   python -m uvicorn inference:app --reload
   ```

2. En la interfaz gráfica (`record.py`), usa el botón **"Grabar y predecir"** para grabar un audio y ver la predicción del modelo ("si" o "no").

## Dependencias

El proyecto requiere los siguientes paquetes de Python (ver `requirements.txt`):

- torch
- torchaudio
- sounddevice
- fastapi
- python-multipart
- requests
- numpy
- scipy
- matplotlib
- scikit-learn
- (tkinter viene incluido en Python estándar para Windows)

## Notas

- Todos los scripts deben ejecutarse desde la carpeta `project`.
- Los audios de entrenamiento deben estar en las carpetas `data/si/` y `data/no/`.
- Puedes agregar más clases (por ejemplo, "unknown") si lo deseas, ajustando el código.
- Si quieres entrenar más épocas, cambia el valor de `num_epochs` en la llamada a `train_model` en `train.py`.

## Contribuyendo

¡Las contribuciones son bienvenidas! Por favor, envía una solicitud de extracción o abre un issue para cualquier mejora o corrección de errores.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.