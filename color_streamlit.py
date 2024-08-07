import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import torch.nn as nn
import numpy as np

class_names = ['fall', 'spring', 'summer', 'winter']

# 모델 로드 함수
def load_model(model_path):
    model = timm.create_model('efficientnet_b4', pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.25020723272413975),  # Dropout 값은 학습된 모델에 맞게 조정
        nn.Linear(num_ftrs, len(class_names))
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 이미지 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(380),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 예측 함수
def predict(image, model):
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return class_names[preds[0]]

# Streamlit UI 구성
st.title('Personal Color')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    with st.spinner('classifying...'):
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        model = load_model('personal_color_efficientnet_b4_76.pth') # 여기에 모델파일명 그대로 입력
        class_names = ['fall', 'spring', 'summer', 'winter']

        image_tensor = preprocess_image(image)
        prediction = predict(image_tensor, model)

        st.subheader(f'Prediction: {prediction}')
