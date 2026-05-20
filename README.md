# 🌿 Plant Disease Detection

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/) 
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.14.0-orange)](https://www.tensorflow.org/) 

A Machine Learning project that identifies **plant diseases** from images using **Convolutional Neural Networks (CNNs)** with **MobileNetV2**. This tool helps farmers and gardeners detect crop diseases for timely intervention.

---

## 🚀 Features

- Detects diseases in multiple plant types.  
- Provides disease description and possible remedies.  
- High accuracy (~95–97% with MobileNetV2).  
- Easy to train with new datasets.  
- Supports real-time predictions.

---

## 🛠️ Technologies Used

- **Python 3.10+** – Main programming language  
- **TensorFlow / Keras** – CNN model building and training
- **Git & GitHub** – Version control  
- **Conda / venv** – Virtual environment management   
- **Streamlit / Flask** – Web app deployment

## 📂 Project Structure
```
plant-disease-identification/src
│
├── app.py
├── utils.py
├── train_model.py
├── PlantDisease/ # Add your dataset folder here
├── model.keras # Trained Keras model
├── requirements.txt
├── .gitignore
├── disease_info.py
└── README.md

```
## ⚙️ Installation & Setup

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```
---
## ▶️ Run the Application

```bash
streamlit run app.py
```
Then open the local URL shown in the terminal (usually http://localhost:8501).
---
## 📊 Dataset

The model is trained on a collection of plant leaf images labeled by plant type and disease.
You can replace the dataset with your own images for retraining or evaluation.


---
