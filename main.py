import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# Setup model dan fungsi pembantu
# =========================================================


import xml.etree.ElementTree as ET


def get_kanji_from_path(path):
    return os.path.basename(os.path.dirname(path))

from utils import load_model, load_gallery, load_kanjidic, l2_norm

# =========================================================
# Streamlit UI
# =========================================================
import styles
styles.apply_custom_style()


st.title("Image Retrieval - Siamese Neural Network (Triplet Loss)")
st.write(
    "Unggah satu gambar, dan sistem akan menampilkan gambar paling mirip dari galeri."
)

embedding_net, device = load_model()
gallery_emb, gallery_paths = load_gallery(embedding_net, device)
kanjidic = load_kanjidic()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

st.sidebar.title("Informasi Sistem")
st.sidebar.write(f"\n\n")
st.sidebar.write("**Model   :** EfficientNet-B0 + Triplet Loss")
st.sidebar.write("**Metode  :** Cosine Similarity")
st.sidebar.write(
    "Semakin skor mendekati nilai 1, skor menunjukkan tingkat kemiripan yang semakin tinggi dengan query."
)
st.sidebar.markdown("---")
st.sidebar.write("Dibuat oleh:  *A.Tamimi Nurrohman*")
st.sidebar.markdown(
    "Silakan isi feedback melalui [form ini](https://docs.google.com/forms/d/e/1FAIpQLSdDmzWpyrQqhxTfrbLSsHzS4HOvb7xbHG5SoIAoSpQ8t6gSaw/viewform?usp=dialog)"
)

uploaded_file = st.file_uploader("Unggah gambar (jpg/png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="Gambar Query", width=200)

    # Dapatkan embedding query
    with torch.no_grad():
        query_tensor = transform(query_img).unsqueeze(0).to(device)
        query_emb = l2_norm(embedding_net(query_tensor)).cpu().numpy()

    # Hitung cosine similarity
    sims = cosine_similarity(query_emb, gallery_emb)[0]
    top_k = 5
    top_indices = np.argsort(sims)[::-1][:top_k]

    st.subheader("Gambar Paling Mirip:")
    cols = st.columns(top_k)

    for i, idx in enumerate(top_indices):
        with cols[i]:
            path = gallery_paths[idx]
            kanji_char = get_kanji_from_path(path)

            info = kanjidic.get(kanji_char, {})

            st.image(path, use_container_width=True)
            st.markdown(f"### {kanji_char}")
            st.write(f"Skor: {sims[idx]:.4f}")

            if info:
                st.write("**Arti:**", ", ".join(info["meanings"][:3]))
                st.write("**Onyomi:**", ", ".join(info["onyomi"]))
                st.write("**Kunyomi:**", ", ".join(info["kunyomi"]))
            else:
                st.write("Data tidak ditemukan")
