import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
import pandas as pd
import smtplib
from datetime import datetime
from email.message import EmailMessage
from keras.models import load_model
from utils.prediction import predict_on_video
from dotenv import load_dotenv
import os

load_dotenv()

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 30
CLASSES_LIST = ["walking", "fight", "running"]

# Load model
model_path = os.path.join("model", "Suspicious_Activity_Detection_LRCN.h5")
model = load_model(model_path)

# CSV log file path
log_path = os.path.join("prediction_logs.csv")
if not os.path.exists(log_path):
    df = pd.DataFrame(columns=["Date", "Time", "Activity", "Confidence"])
    df.to_csv(log_path, index=False)

# Email Alert Function
def send_email_alert(activity, confidence):
       EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")

    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        st.error("❌ Email credentials not found. Check .env file.")
        return
        
    msg = EmailMessage()
    msg["Subject"] = "🚨 Suspicious Activity Detected!"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = RECEIVER_EMAIL
    msg.set_content(f"Alert: {activity.upper()} detected with {confidence*100:.2f}% confidence.")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
            st.success("📧 Email alert sent!")
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")

# Streamlit UI
st.set_page_config(page_title="🔍 Suspicious Activity Detector", layout="centered")
st.markdown("""
    <h1 style='text-align: center;'>🎥 Suspicious Activity Detection System</h1>
    <hr style='border:1px solid #f5f5f5;'>
    <div style='text-align:center;'>Upload video, run prediction, and get alerted in real-time!</div>
    <br>
""", unsafe_allow_html=True)

option = st.sidebar.radio("📂 Select Option:", ["1. Upload and Predict", "2. View Logs", "4. Report Suspicious Video", "5. Performance Stats"])

if option == "1. Upload and Predict":
    uploaded_file = st.file_uploader("📁 Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name

        st.video(video_path)
        st.info("⏳ Analyzing the video. Please wait...")

        start_time = time.time()

        output_filename = f"output_{int(time.time())}.avi"
        output_video_path = os.path.join("output_videos", output_filename)

        predicted_class, confidence = predict_on_video(video_path, output_video_path, model, SEQUENCE_LENGTH, CLASSES_LIST, IMAGE_HEIGHT, IMAGE_WIDTH)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        inference_time = end_time - start_time

        if predicted_class == "Video too short":
            st.error("❌ Video too short for prediction. Please upload a longer video.")
        else:
            st.success(f"✅ Detected Activity: **{predicted_class.upper()}** with {confidence*100:.2f}% confidence")
            st.video(output_video_path)

            if predicted_class == "fight":
                send_email_alert(predicted_class, confidence)

            now = datetime.now()
            log_entry = pd.DataFrame([[now.date(), now.strftime('%H:%M:%S'), predicted_class, confidence]],
                                     columns=["Date", "Time", "Activity", "Confidence"])
            log_entry.to_csv(log_path, mode='a', index=False, header=False)

            with st.expander("📊 Show Performance Stats"):
                st.metric("⏱ Inference Time (s)", f"{inference_time:.2f}")
                st.metric("⚡ FPS", f"{fps:.2f}")

elif option == "2. View Logs":
    st.subheader("🧾 Prediction Logs")
    if os.path.exists(log_path):
        logs = pd.read_csv(log_path)
        st.dataframe(logs)
    else:
        st.warning("No logs found.")

elif option == "4. Report Suspicious Video":
    st.subheader("📤 Report a Video")
    st.write("Use this to manually report a suspicious video.")
    suspicious_file = st.file_uploader("📁 Upload Suspicious Video", type=["mp4", "avi", "mov"])
    if suspicious_file is not None:
        st.success("📤 Video reported successfully. (Feature coming soon to send to authorities)")

elif option == "5. Performance Stats":
    st.subheader("📈 Inference Performance Overview")
    if os.path.exists(log_path):
        logs = pd.read_csv(log_path)
        st.line_chart(logs[["Confidence"]])
        st.bar_chart(logs["Activity"].value_counts())
    else:
        st.info("No logs to show yet.")
