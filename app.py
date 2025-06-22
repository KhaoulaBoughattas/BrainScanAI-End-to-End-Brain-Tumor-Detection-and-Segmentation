import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


model = BrainTumorClassifier(num_classes=4)
model.load_state_dict(torch.load("model/model_multitask.pth", map_location=torch.device('cpu')))
model.eval()


st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title(" Brain Tumor Classification")
st.write("Upload an MRI image and get predictions about tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, 1)

    st.success(f"###  Predicted: {class_names[pred_class.item()]} \nConfidence: {confidence.item()*100:.2f}%")

    st.subheader("Confidence for Each Class")
    st.bar_chart({class_names[i]: probs[0][i].item() for i in range(len(class_names))})
