# CheXpert Chest X-ray Classification Streamlit App

This repository provides a production-ready Streamlit web application deployment for a medical image classification project (CheXpert) using **TensorFlow** and **EfficientNetB0**.

## Project Structure

- `app.py`: High-fidelity, robust Streamlit UI featuring image upload, results visualization with custom bar charts, pathology scorecards, threshold configuration, and comprehensive error handling.
- `utils.py`: Contains core image preprocessing logic (converting to RGB, resizing to $224 \times 224 \times 3$), EfficientNetB0 model reconstruction, weight loading, model prediction, and system logging.
- `requirements.txt`: List of dependencies compatible with Python 3.11.9.
- `feature.weights.h5`: Trained model weights file containing the weights for the model layers.

---

## Deployment & Setup Guide

Please follow these steps to deploy and run the application locally or on a server.

### 1. Create a Virtual Environment

Initialize a clean virtual environment using **Python 3.11.9**.

On Windows:
```powershell
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

Install the required packages using the precise requirements list:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Place Model Weights File

Ensure that your weights file (`feature.weights.h5`) is placed in the root directory. Alternatively, you can configure the path to the weights file directly in the Streamlit application sidebar.

### 4. Run the Streamlit Application

Start the web application locally by running:
```bash
streamlit run app.py
```

After executing the command, the portal will be accessible at:
- **Local URL:** `http://localhost:8501`
- **Network URL:** `http://<your-network-ip>:8501`

---

## Technical Specifications

- **Model backbone**: EfficientNetB0
- **Input resolution**: $224 \times 224 \times 3$
- **Pathologies classified (14 classes)**:
  1. No Finding
  2. Enlarged Cardiomediastinum
  3. Cardiomegaly
  4. Lung Opacity
  5. Lung Lesion
  6. Edema
  7. Consolidation
  8. Pneumonia
  9. Atelectasis
  10. Pneumothorax
  11. Pleural Effusion
  12. Pleural Other
  13. Fracture
  14. Support Devices

---

## Troubleshooting & Logging

The application implements PEP 8 compliant logging to debug inference and model loading. Detailed logs will appear in the console while running.
- If model loading fails, verify the file path of your weights file `feature.weights.h5`.
- If image preprocessing fails, check that the uploaded file is a valid image (PNG, JPG, or JPEG) and not corrupted.
