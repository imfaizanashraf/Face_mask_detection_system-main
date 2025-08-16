import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model/model.pt')
IMG_SIZE = 224

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def pspi_to_class(pspi):
    if pspi < 0.5:
        return "No pain"
    elif pspi < 1.5:
        return "Very mild"
    elif pspi < 2.5:
        return "Mild"
    elif pspi < 3.5:
        return "Moderate"
    elif pspi < 4.5:
        return "Moderately severe"
    elif pspi < 5.5:
        return "Severe"
    else:
        return "Very severe"

st.title('Pain Recognition from Facial Expression')
st.write('Upload a facial image to predict pain intensity (PSPI score and class).')

uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('Predicting...')
    model = load_model()
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        pain_score = output.item()
        pain_class = pspi_to_class(pain_score)
    st.success(f'Predicted Pain Intensity: {pain_class} (PSPI: {pain_score:.2f})')
