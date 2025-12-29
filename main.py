import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# Setup model dan fungsi pembantu
# =========================================================

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "triplet_epoch10.pth"

    # Definisikan arsitektur sama persis seperti di notebook
    backbone = models.efficientnet_b0(pretrained=True)
    backbone.classifier = nn.Identity()
    embedding_net = nn.Sequential(
        backbone,
        nn.Linear(1280, 128)
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    embedding_net.load_state_dict(checkpoint["model_state"])
    embedding_net.eval()
    return embedding_net, device


def l2_norm(x):
    return x / x.norm(p=2, dim=1, keepdim=True)


@st.cache_resource
def load_gallery(_embedding_net, device):
    gallery_dir = "gallery"
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    gallery_images, gallery_paths = [], []
    for root, _, files in os.walk(gallery_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(root, file)
                img = Image.open(img_path).convert("RGB")
                gallery_images.append(transform(img))
                gallery_paths.append(img_path)

    gallery_tensor = torch.stack(gallery_images).to(device)

    with torch.no_grad():
        gallery_emb = l2_norm(_embedding_net(gallery_tensor)).cpu().numpy()

    return gallery_emb, gallery_paths



# =========================================================
# Streamlit UI
# =========================================================
st.markdown("""
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
""", unsafe_allow_html=True)


st.title("Image Retrieval - Siamese Neural Network (Triplet Loss)")
st.write("Unggah satu gambar, dan sistem akan menampilkan gambar paling mirip dari galeri.")

embedding_net, device = load_model()
gallery_emb, gallery_paths = load_gallery(embedding_net, device)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.sidebar.title("Informasi Sistem")
st.sidebar.write(f"\n\n")
st.sidebar.write(f"**Device :** {device}")
st.sidebar.write("**Model   :** EfficientNet-B0 + Triplet Loss")
st.sidebar.write("**Metode  :** Cosine Similarity")
st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Dibuat oleh:  *A.Tamimi Nurrohman*")

uploaded_file = st.file_uploader("Unggah gambar (jpg/png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="Gambar Query", use_container_width=True)

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
            st.image(gallery_paths[idx],
                     caption=f"Skor: {sims[idx]:.4f}",
                     use_container_width=True)
