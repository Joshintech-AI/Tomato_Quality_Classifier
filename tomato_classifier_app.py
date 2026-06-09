import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# --- Configuration and Hyperparameters (Match your Colab notebook settings) ---
IMAGE_SIZE = 64
num_classes_s1 = 3  # For 'Damaged', 'ripe', 'unripe'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names for Stage 1
class_names_s1 = ['Damaged', 'ripe', 'unripe']

# --- Model Definition (TomatoCNN - must be identical to training) ---
class TomatoCNN(nn.Module):
    def __init__(self, num_classes=3, image_size=64):
        super(TomatoCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        flat_features = 128 * (image_size // 8) * (image_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Image Transformations for Inference ---
val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# --- Prediction Function ---
def predict_image(image_file, model, class_names):
    img = Image.open(image_file).convert('RGB')
    img_tensor = val_test_transforms(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    return predicted_class, confidence_pct, img

# --- Streamlit Application UI ---
st.set_page_config(layout="centered", page_title="Tomato Classifier", page_icon="🍅")

# Sidebar content
st.sidebar.header("About This App")
st.sidebar.write(
    "This application uses a Convolutional Neural Network (CNN) to classify tomato images."
    "It can identify if a tomato is **ripe**, **unripe**, or **damaged**."
)
st.sidebar.image("Tomato Gif.gif", width="stretch")
st.sidebar.markdown("--- ")
st.sidebar.header("How to Use")
st.sidebar.info(
    "1. Upload an image of a tomato using the file uploader.\n"
    "2. The model will then predict its condition.\n"
    "3. Get instant recommendations!"
)

st.title("🍅 Tomato Quality Classifier")
st.markdown("Upload an image of a tomato to classify its state.")

# Load the model
@st.cache_resource
def load_model():
    model = TomatoCNN(num_classes=num_classes_s1, image_size=IMAGE_SIZE)
    # IMPORTANT: Ensure 'best_model_stage1.pth' is in the same directory as this script
    model_path = "best_model_stage1.pth"
    if not os.path.exists(model_path):
        st.error(f"Model weights file '{model_path}' not found. Please ensure it's in the same directory as this Streamlit app.")
        st.stop()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model_s1 = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width="stretch")
    st.write("")
    st.write("Classifying...")

    # Display a spinner while classifying
    with st.spinner('Analyzing the tomato...'):
        predicted_class, confidence_pct, original_img = predict_image(uploaded_file, model_s1, class_names_s1)

    st.success(f"Prediction: **{predicted_class}** with **{confidence_pct:.2f}%** confidence")

    # Display recommendations based on prediction
    st.subheader("Recommendation:")
    if predicted_class == 'ripe':
        st.write("This tomato is **ripe**! It's healthy and ready for consumption. Enjoy!")
        st.balloons() # Add a small celebration
    elif predicted_class == 'unripe':
        st.write("This tomato is **unripe**. You can store it to ripen further at room temperature.")
    elif predicted_class == 'Damaged':
        st.write("This tomato appears **damaged**. It may not be suitable for consumption and could be returned or discarded. Consider composting!")
    else:
        st.write("No specific recommendation for this class.")

st.markdown("--- ")
st.markdown("Developed by TEAM JARLS")
