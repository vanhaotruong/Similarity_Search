
import streamlit as st
import os


def image_file_path(uploaded_files):
    temp_folder = "predict_image"
    os.makedirs(temp_folder, exist_ok=True)
    saved_paths = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)

    return saved_paths
