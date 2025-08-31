import streamlit as st
import os
import shutil
import subprocess
from pathlib import Path
import requests

# Ensure folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Download YOLOv5 model if missing
if not os.path.exists("yolov5s.pt"):
    st.write("Downloading YOLOv5 model...")
    url = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt"
    r = requests.get(url, stream=True)
    with open("yolov5s.pt", "wb") as f:
        shutil.copyfileobj(r.raw, f)

# Download DeepSort ckpt if missing
ckpt_path = "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
if not os.path.exists(ckpt_path):
    st.write("Downloading DeepSort checkpoint...")
    url = "https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/releases/download/v1.0/ckpt.t7"
    r = requests.get(url, stream=True)
    with open(ckpt_path, "wb") as f:
        shutil.copyfileobj(r.raw, f)

st.title("🚀 Object Tracking with YOLOv5 + DeepSORT")

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded video
    video_path = os.path.join("uploads", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.video(video_path)

    # Output video path
    output_path = os.path.join("outputs", f"tracked_{uploaded_file.name}")

    # Run detection + tracking
    st.write("🔄 Running object tracking, please wait...")
    try:
        command = [
            "python", "detect_sort.py",
            "--weights", "yolov5s.pt",
            "--source", video_path,
            "--device", "cpu",
            "--conf-thres", "0.5",
            "--iou-thres", "0.45",
            "--classes", "0",   # person class
            "--project", "outputs",
            "--name", f"tracked_{Path(uploaded_file.name).stem}",
            "--exist-ok"
        ]
        subprocess.run(command, check=True)

        # Show result
        result_video = f"outputs/tracked_{Path(uploaded_file.name).stem}/{uploaded_file.name}"
        if os.path.exists(result_video):
            st.success("✅ Tracking complete!")
            st.video(result_video)

            with open(result_video, "rb") as f:
                st.download_button(
                    label="Download Tracked Video",
                    data=f,
                    file_name=f"tracked_{uploaded_file.name}",
                    mime="video/mp4"
                )
        else:
            st.error("⚠️ Tracking failed: No result video found.")

    except Exception as e:
        st.error(f"⚠️ Error during tracking: {str(e)}")
