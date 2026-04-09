import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
import torch.nn as nn
import streamlit as st
from torchvision import models, transforms

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "triplet_epoch15.pth"

    # Definisikan arsitektur sama persis seperti di notebook
    backbone = models.efficientnet_b0(pretrained=True)
    backbone.classifier = nn.Identity()
    embedding_net = nn.Sequential(backbone, nn.Linear(1280, 128)).to(device)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    embedding_net.load_state_dict(checkpoint["model_state"])
    embedding_net.eval()
    return embedding_net, device


def l2_norm(x):
    return x / x.norm(p=2, dim=1, keepdim=True)

@st.cache_resource
def load_kanjidic():
    tree = ET.parse("kanjidic2.xml")
    root = tree.getroot()

    kanji_dict = {}

    for char in root.findall("character"):
        literal = char.find("literal").text

        meanings = []
        onyomi = []
        kunyomi = []

        for rm in char.findall(".//rmgroup"):
            for m in rm.findall("meaning"):
                if m.get("m_lang") is None:
                    meanings.append(m.text)

            for r in rm.findall("reading"):
                if r.get("r_type") == "ja_on":
                    onyomi.append(r.text)
                elif r.get("r_type") == "ja_kun":
                    kunyomi.append(r.text)

        kanji_dict[literal] = {
            "meanings": meanings,
            "onyomi": onyomi,
            "kunyomi": kunyomi,
        }

    return kanji_dict

@st.cache_resource
def load_gallery(_embedding_net, device):
    gallery_dir = "gallery"
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

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