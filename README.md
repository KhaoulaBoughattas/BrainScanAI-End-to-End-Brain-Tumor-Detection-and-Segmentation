#  BrainScanAI: End-to-End Brain Tumor Detection and Segmentation

This project provides a complete, explainable, and interactive AI-based pipeline for brain tumor detection from MRI images. Everything is implemented in a **single Jupyter Notebook** — from classification to segmentation to Grad-CAM visualizations and user interaction.

---

##  Features

✅ Brain Tumor **Classification** (Glioma, Meningioma, Pituitary, No Tumor)  
✅ Tumor **Segmentation** (UNet-based)  
✅ Visual **Explanations** using Grad-CAM  
✅ **Confidence Scores** for predictions  
✅ **Test-Time Augmentation** (TTA)  

---

##  Contents

- `project.ipynb` – Main notebook with all models and visualizations
- `models/` –  Pretrained models
- `data/` – MRI images for testing and validation
- `requirements.txt` – Python dependencies
- `README.md` – You're reading it!

---

##  Classification

We trained and evaluated multiple deep learning models:
- `ConvNeXt`, `EfficientNet`, `ResNet18`, etc.
- Achieved up to **95% accuracy** on test data.
- Final predictions include a **confidence score** and **Grad-CAM heatmap**.

---

##  Segmentation

UNet-based architecture for pixel-wise segmentation of tumor regions:
- Output: binary mask over the original MRI
- Visualized using color overlays

 Dice Score ≈ **0.89**

---

##  Test-Time Augmentation (TTA)

Improve model robustness using test-time transformations:
- Horizontal flips
- Rotation
- Cropping  
 Final prediction is the average of TTA runs

---

##  Confidence Scores

Softmax-based confidence values are shown for each predicted class to help trust the model decision.

---

##  Interactive App 

You can launch an interface (Flask) to upload an MRI and:
- Classify the tumor
- Segment it
- Show Grad-CAM & confidence

