# 🚨 Suspicious Activity Detection System

A **real-time Suspicious Activity Detection System** built using **Deep Learning (LRCN model)** and **Streamlit**, designed to analyze video footage and automatically detect activities such as **walking, running, and fighting**.

This project demonstrates how computer vision and sequence-based neural networks can be used for intelligent video surveillance and safety monitoring.

---

## 📌 About the Project

The **Suspicious Activity Detection System** takes a video as input and processes it frame-by-frame to identify suspicious human activities.  
It uses a **Long-term Recurrent Convolutional Network (LRCN)**, which combines:

- **CNN (Convolutional Neural Networks)** for spatial feature extraction  
- **LSTM (Long Short-Term Memory)** for temporal sequence learning  

The system is capable of:
- Detecting activities from video clips
- Highlighting suspicious actions (e.g., fighting)
- Sending **email alerts** when dangerous activity is detected
- Logging predictions for performance analysis

---

## 🎯 Key Features

- 🎥 Upload video files (`.mp4`, `.avi`, `.mov`)
- 🤖 Deep learning–based activity classification
- 📧 Email alert system for suspicious activities
- 📊 Performance statistics (confidence, inference time, FPS)
- 🧾 Prediction logging for analysis
- 🖥️ Interactive web interface using Streamlit

---

## 🧠 Activities Detected

- 🚶 Walking  
- 🏃 Running  
- ⚠️ Fighting (treated as suspicious activity)

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy & Pandas**
- **Streamlit**
- **SMTP (Email alerts)**

---

## 📂 Project Structure

```text
Suspicious-Activity-Detection/
│── app.py
│── utils/
│   └── prediction.py
│── requirements.txt
│── .gitignore
│── README.md
