import streamlit as st

def apply_custom_style():
    st.markdown(
        """
    <style>
    /* Background dan warna utama */
    .main {
        background-color: #f9fafc;
        font-family: 'Poppins', sans-serif;
    }

    /* Judul utama */
    h1 {
        color: #2E86C1;
        text-align: center;
        font-weight: 700;
    }

    /* Subtitle */
    h2, h3, h4 {
        color: #34495E;
        font-weight: 600;
    }

    /* File uploader */
    .stFileUploader {
        border: 2px dashed #2E86C1;
        border-radius: 10px;
        background-color: #ffffff;
        padding: 1em;
        text-align: center;
    }

    /* Gambar hasil */
    .stImage img {
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.2s ease;
    }
    .stImage img:hover {
        transform: scale(1.02);
    }

    /* Tombol */
    div.stButton > button:first-child {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #1A5276;
    }

    /* Spinner dan hasil */
    .css-1y0tads {
        text-align: center;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )