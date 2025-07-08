# 🌱 Plant Pathology AI - Streamlit App

[![Streamlit App](https://img.shields.io/badge/Try%20it%20yourself-Live%20Demo-brightgreen?logo=streamlit)](https://plantpathology-8bawdpclz4zwd56v46tstt.streamlit.app/)

---

## 🚀 Try it yourself - Live Deployed Link in Streamlit 

👉 **[Live Deployed App](https://plantpathology-8bawdpclz4zwd56v46tstt.streamlit.app/)**

---

## 🧑‍🔬 Project Overview

This application leverages the **Plant Pathology 2020 Challenge dataset** from Cornell University to provide accurate disease classification for apple leaves. The model uses traditional machine learning techniques with handcrafted color histogram features and Random Forest classification, all wrapped in a beautiful and interactive Streamlit web app.

---

## ✨ Features
- 🌿 Multi-class disease classification (Healthy, Rust, Scab, Multiple Diseases)
- 🔬 Handcrafted feature extraction (color histograms)
- 🌲 Random Forest classifier for robust predictions
- 📊 Probability/confidence scores for each class
- 🖼️ Upload your own leaf images for instant results
- 🎨 Modern, professional, and interactive UI
- ☁️ **Live demo available!**

---

## 🛠️ Tech Stack
- **Python 3.8+**
- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [OpenCV (opencv-python-headless)](https://pypi.org/project/opencv-python-headless/)
- [Pandas, NumPy, Pillow, Matplotlib, scikit-image, joblib]

---

## 🖥️ Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ankit-03G/Plant_Pathology.git
   cd Plant_Pathology
   ```
2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optional) Train the model:**
   - Run `train_model.py` to generate the model and scaler if not already present.
   ```bash
   python train_model.py
   ```
5. **Run the app locally:**
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deployment

This app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). You can deploy your own version by:
- Forking this repo
- Connecting your GitHub to Streamlit Cloud
- Setting `app.py` as the main file
- Ensuring `requirements.txt` is up to date

---

## 📂 Project Structure
```
Plant_Pathology/
├── app.py                # Streamlit web app
├── train_model.py        # Model training script
├── requirements.txt      # Python dependencies
├── models/               # Saved model & scaler (.joblib)
├── data/                 # (Optional) Data files/images
├── .streamlit/           # Streamlit config
└── README.md             # This file
```

---

## 📬 Contact

- [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/ankit-kumar-gupta-6ba724266/)
- [![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/Ankit-03G)
- 📧 ankitkumargupta030204@gmail.com

---

> Made with ❤️ by Ankit Kumar Gupta 
