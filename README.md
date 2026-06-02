# IAfishingDetector

> **Estructura por ubicaciones (Etapa 9A):** cada lugar de pesca tiene su modelo y su ROI en
> `locations/<ubicación>/` (modelo `.onnx` gitignored, `roi.yaml`/`NOTES.md` versionados). El modelo se
> elige en `config.yaml` con `detector_mode` (general|specific) y `location`. Ver **`locations/README.md`**.
> El bot es 100% visual (trigger por foam); el audio fue eliminado del loop en la Etapa 8C.

Un proyecto de detección y automatización para la pesca en World of Warcraft 4.3.4.  
Combina reconocimiento de audio y visión por computadora (YOLOv5) para:  
- Detectar el sonido de la caña al pescar.  
- Localizar el corcho en pantalla.  
- Mover el cursor y hacer clic automáticamente.  

---

## 📁 Estructura del repositorio

```
IAfishingDetector/
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── detect.py
├── sound_validator.py
├── capturador.py
├── normalize_sounds.py
├── compare_fft.py
├── compare_live_sound.py
├── compare_live_sound_fft.py
├── train_yolo.py
└── yolov5s.pt
```

- **main.py**  
  Orquesta el ciclo audio→visión→clic.  
- **detect.py**  
  Detecta el corcho en pantalla usando el modelo YOLO.  
- **sound_validator.py**  
  Valida y preprocesa los archivos de audio de referencia.  
- **capturador.py**  
  Captura audio en tiempo real desde el sistema.  
- **normalize_sounds.py**  
  Normaliza niveles y formato de audio para entrenamiento.  
- **compare_fft.py**, **compare_live_sound_fft.py**  
  Comparan señales en el dominio frecuencia (FFT).  
- **compare_live_sound.py**  
  Comparación directa de formas de onda (cross‑correlation).  
- **train_yolo.py**  
  Script para entrenar tu propio modelo YOLO con tus imágenes.  
- **yolov5s.pt**  
  Peso base de YOLOv5 (light).  

> **Nota:** Los directorios `dataset/`, `labelimg-env/` y `normalized/` están en `.gitignore` y no se versionan.

---

## 🚀 Requisitos

- Python 3.10+  
- `ffmpeg` (para procesado de audio)  
- Entorno virtual (recomendado)  

---

## ⚙️ Instalación

1. **Clona el repositorio** y entra en la carpeta:
   ```bash
   git clone https://github.com/Leo921007/IAfishingDetector.git
   cd IAfishingDetector
   ```
2. **Crea y activa un entorno virtual**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```
3. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Verifica que `ffmpeg` esté en tu PATH**:
   ```bash
   ffmpeg -version
   ```

---

## 🎯 Uso

1. **Validar sonidos de referencia**  
   ```bash
   python sound_validator.py --input Fishing_1.wav Fishing_2.wav Fishing_3.wav
   ```
2. **Normalizar audios**  
   ```bash
   python normalize_sounds.py --input Fishing_*.wav --output normalized/
   ```
3. **Comparar señal en vivo**  
   - Dominio tiempo:
     ```bash
     python compare_live_sound.py --reference normalized/Fishing_1.wav
     ```
   - Dominio frecuencia:
     ```bash
     python compare_live_sound_fft.py --reference normalized/Fishing_1.wav
     ```
4. **Detectar corcho y clic automático**  
   ```bash
   python main.py
   ```
5. **Entrenar tu modelo YOLO**  
   Prepara tu dataset (imágenes + labels) y luego:
   ```bash
   python train_yolo.py --data dataset/data.yaml --cfg yolov5s.yaml --epochs 50
   ```

---

## 🔧 Configuración

- Ajusta umbrales de detección en `detect.py` y `compare_live_sound.py`.  
- Cambia el modelo o ruta en `train_yolo.py` según tu configuración de YOLOv5.  

---

## 📚 Referencias

- [YOLOv5](https://github.com/ultralytics/yolov5)  
- Documentación oficial de `pyaudio`, `opencv-python`, `numpy`, etc.  

---

## 📝 Licencia

Proyecto bajo licencia MIT. Consulta el archivo `LICENSE` para más detalles.