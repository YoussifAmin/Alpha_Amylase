# Alpha_Amylase

# 🧪 Alpha_Amylase: Point-of-Care Classification from Saliva Images

This project implements a **deep learning pipeline** for classifying salivary α-amylase concentration levels using a YOLO-based segmentation model followed by post-classification logic. It includes a trained model, a test image set, label maps, and evaluation scripts.

---

## 📁 Project Structure
Alpha_Amylase/
├── 1_predict.py # Runs the trained YOLO model on test images
├── 2_Checkk_accuracy.py # Computes accuracy based on predicted vs. actual labels
├── Model.pt # Pretrained YOLOv8 model
├── labelmap.txt # Maps class IDs to RGB colors
├── Testset/ # Test images to classify
├── overlay/ # YOLO-processed images with predicted labels
├── mask/ # Segmentation masks
└── correct_labels/ # Ground-truth class folders


---



## 🧠 Step 1 – Predict α-Amylase Levels

Run: python 1_predict.py
```

This script will:
- Load `Model.pt` (YOLOv8 segmentation)
- Run predictions on all images in `Testset/`
- Generate overlays and masks in `overlay/` and `mask/`

---

## ✅ Step 2 – Evaluate Accuracy

Run: python 2_Checkk_accuracy.py
```

This script:
- Compares predictions with true labels from `correct_labels/`
- Print accuracy
- Displays confusion matrix (counts and percentages)
