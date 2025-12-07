# Pi5 Fun Text Interpreter

This project demonstrates how to build a fun text interpreter on Raspberry Pi 5 using ERNIE-4.5-0.3B. Use this repository as a minimal demo to quickly prototype a hardware project based on ERNIE-4.5-0.3B.

This project mainly covers:
- Running ERNIE-4.5-0.3B on Raspberry Pi 5 via llama.cpp
- Performing photo-based OCR and fun text interpretation on Raspberry Pi 5

> Because Raspberry Pi 5's system-level Python package management may not allow installing global dependencies directly, you should use a virtual environment to manage project dependencies. Create and activate a virtual environment first:
```bash
python -m venv venv
source venv/bin/activate
```

![alt text](hardware.jpg)

## Install and deploy ERNIE-4.5-0.3B
> Nearly all end-side devices can run ERNIE-4.5-0.3B using llama.cpp. 

### Install dependencies
Installing llama-cpp-python via pip may take a long time. If your hardware is low-end, consider using brew or building from source.
```bash
pip install llama-cpp-python uvicorn anyio starlette fastapi sse_starlette starlette_context pydantic_settings
```

### Prepare gguf model file
llama.cpp requires models in gguf format. You can download ERNIE-4.5-0.3B gguf files from Hugging Face or ModelScope.

```bash
# Download from Hugging Face
pip install -U "huggingface_hub[cli]"
hf download unsloth/ERNIE-4.5-0.3B-PT-GGUF ERNIE-4.5-0.3B-PT-Q4_K_M.gguf --local-dir .

# Download from ModelScope
pip install -U "modelscope"
modelscope download --model unsloth/ERNIE-4.5-0.3B-PT-GGUF ERNIE-4.5-0.3B-PT-Q4_K_M.gguf --local_dir .
```

### Start an OpenAI-compatible API server
```bash
python -m llama_cpp.server --model ERNIE-4.5-0.3B-PT-Q4_K_M.gguf --host 0.0.0.0
```

### Call the service
You can call the server via an OpenAI-compatible client. Open a new terminal, reactivate the virtual environment and install dependencies:
```bash
source venv/bin/activate
pip install openai
```

Example usage:
```python
import openai

server_url = "http://localhost:8000/v1"

client = openai.OpenAI(
    base_url=server_url,
    api_key="anyvalue"
)

response = client.chat.completions.create(
    model="anyvalue",
    messages=[
        {"role": "user", "content": "hi, how are you?"}
    ],
    temperature=0.7,
    max_tokens=500
)

assistant_reply = response.choices[0].message.content
print(assistant_reply)
```

## Build the Fun Text Interpreter
### Install OCR tools
This project uses RapidOCR as the OCR tool. RapidOCR is an open-source OCR tool based on [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), which is Baidu's PaddlePaddle open-source OCR library supporting multi-language text recognition with high accuracy and efficiency. Activate the virtual environment in a new terminal and install dependencies:
```bash
source venv/bin/activate
pip install rapidocr onnxruntime
```

Place an image named `input.jpg` in the project root to test OCR. Run:
```python
from rapidocr import EngineType, LangDet, ModelType, OCRVersion, RapidOCR

engine = RapidOCR(
    params={
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV4
    }
)

img_url = "input.jpg"
result = engine(img_url)

print(result.txts)
```

### Wrap OCR as a service
To facilitate usage, wrap OCR into a REST API:
```python
# ocr_server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from rapidocr import EngineType, ModelType, OCRVersion, RapidOCR
import uvicorn
import os
from typing import List

# Initialize OCR engine
ocr_engine = RapidOCR(
    params={
        "Det.engine_type": EngineType.ONNXRUNTIME,
        "Det.model_type": ModelType.MOBILE,
        "Det.ocr_version": OCRVersion.PPOCRV4,
    }
)

app = FastAPI(title="RapidOCR Service")

@app.post("/ocr")
async def ocr_service(image: UploadFile = File(...)) -> dict:
    """
    Accepts an uploaded image and returns OCR results (list of strings)
    - Input: uploaded image file (PNG/JPG/JPEG supported)
    - Output: {"result": ["text1", "text2", ...]}
    """
    try:
        # Validate file type
        if image.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Only PNG/JPG/JPEG images are supported")
        
        # Save temporary file
        temp_path = f"temp_{image.filename}"
        with open(temp_path, "wb") as f:
            f.write(await image.read())
        
        # Run OCR
        ocr_result = ocr_engine(temp_path)
        texts = ocr_result.txts
        
        # Clean up
        os.remove(temp_path)
        
        return {"result": texts}
    
    except Exception as e:
        # Remove temp file on error if it exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
```

Run the OCR service:
```bash
python ocr_server.py
```

### Capture images from the Pi5 CSI camera (or USB camera)
The CSI camera is compact and suitable for embedded use. The following is for Raspberry Pi 5; these steps should be run outside the virtual environment in a new terminal.

Install camera dependency:
```bash
sudo apt install python3-picamera2
```

Example using Picamera2:
```python
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.start()  # Must start the camera (even without preview)
picam2.capture_file("test_picam2.jpg")  # Capture
picam2.stop()
print("Photo saved as test_picam2.jpg")
```

If using a USB camera, use OpenCV:
```python
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, frame = cap.read()
if ret:
    cv2.imwrite("usb_camera_image.jpg", frame)
    print("Image saved")
else:
    print("Capture failed")

cap.release()
```

### Build the Fun Text Interpreter UI
Install PyQt5 and build a simple UI to continuously show the camera stream and trigger capture + OCR + fun text interpretation on click.

Note: On Raspberry Pi 5, you may need to install PyQt5 outside the virtual environment:
```bash
sudo apt install python3-pyqt5
```

Save the following as `show.py`:
```python
import sys
import requests
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, 
    QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from picamera2 import Picamera2
from io import BytesIO
from PIL import Image

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raspberry Pi Camera + OCR + Fun Text Interpreter")
        self.resize(800, 600)
        
        # Initialize camera (RGB)
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        self.picam2.configure(camera_config)
        self.picam2.start()
        
        # UI
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)
        
        # Timer to update preview
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
    
    def mousePressEvent(self, event):
        """On mouse click -> capture and interpret text funnily"""
        if event.button() == Qt.LeftButton:
            self.capture_and_translate()
    
    def update_frame(self):
        frame = self.picam2.capture_array()
        if frame is not None:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.label.width(), self.label.height(), Qt.KeepAspectRatio
            )
            self.label.setPixmap(pixmap)
    
    def capture_and_translate(self):
        try:
            # 1. Capture and call OCR service
            frame = self.picam2.capture_array()
            pil_image = Image.fromarray(frame)
            img_buffer = BytesIO()
            pil_image.save(img_buffer, format="JPEG")
            img_buffer.seek(0)
            
            ocr_url = "http://localhost:8001/ocr"  # OCR service endpoint
            ocr_response = requests.post(
                ocr_url, 
                files={"image": ("capture.jpg", img_buffer, "image/jpeg")}
            )
            
            if ocr_response.status_code != 200:
                raise Exception(f"OCR service error: {ocr_response.text}")
            
            ocr_text = ocr_response.json()["result"][0]  # take first line
            
            # 2. Call llama.cpp interpretation via OpenAI-compatible API
            translate_url = "http://localhost:8000/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-dummy"
            }
            data = {
                "model": "ernie-4.5-0.3b",  # Use ERNIE-4.5-0.3B model
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a fun assistant. Explain the content of the following text in a fun way."
                    },
                    {
                        "role": "user",
                        "content": f"Please explain the following text in a fun way: {ocr_text}"
                    }
                ]
            }
            
            translate_response = requests.post(
                translate_url, 
                headers=headers, 
                json=data
            )
            
            if translate_response.status_code != 200:
                raise Exception(f"Interpretation service error: {translate_response.text}")
            
            # 3. Parse interpretation
            translation = translate_response.json()["choices"][0]["message"]["content"]
            
            # 4. Show original and interpreted text
            result_text = f"【Original】{ocr_text}\n\n【Fun Interpretation】{translation}"
            QMessageBox.information(self, "Fun Text Interpretation Result", result_text)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed: {str(e)}")
    
    def closeEvent(self, event):
        self.timer.stop()
        self.picam2.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec())
```


Run `show.py` to start the app. Ensure the OCR service and the llama.cpp server are running beforehand. After running, a camera preview will appear — point it at text and click to trigger capture, OCR, and fun text interpretation.

## Future Outlook

This project demonstrates the feasibility of building a fun text interpreter on Raspberry Pi 5 using the ERNIE-4.5-0.3B model. As a minimal demo, it inspires end-side AI applications. In the future, we can expand and improve in the following areas:

### Model Optimization
- **Parameter Upgrade**: Try using larger ERNIE models (e.g., ERNIE-3.5-8B or higher) to improve text interpretation accuracy and fun factor.
- **Model Fine-tuning**: Fine-tune for specific domains (e.g., education, entertainment) to make interpretations more tailored.
- **Quantization Optimization**: Further optimize model quantization (e.g., using GGUF Q2_K or lower precision) for lower-power devices.

### Feature Expansion
- **Multi-language Support**: Integrate multi-language OCR and translation for global users.
- **Voice Interaction**: Add voice recognition input and speech synthesis output for full voice interaction.
- **Enhanced Interaction**: Support continuous capture, batch processing, or user feedback to improve interpretation quality.
- **Cloud Collaboration**: Collaborate with cloud large models when network is available to enhance interpretation.

### Hardware Adaptation
- **More Devices**: Adapt to other edge devices with compute acceleration to leverage hardware acceleration (e.g., GPU, TPU) for faster inference and lower latency.

### Application Scenarios
- **Educational Tools**: For children's learning, helping understand complex text.
- **Entertainment Apps**: Provide fun interpretations in games or social settings.
- **Assistive Tools**: Help visually impaired people understand text or language learners practice reading.

We encourage the open-source community to contribute. If you have ideas or suggestions, feel free to submit Issues or Pull Requests!

---

*This document is based on the PaddlePaddle community open-source project, aiming to promote the popularization and application of AI technology.*
