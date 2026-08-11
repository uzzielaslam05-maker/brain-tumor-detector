# Brain Tumor Detector — Astrocytoma & Glioblastoma Detection

A Streamlit web app that uses a YOLOv8 model to detect signs of Astrocytoma or Glioblastoma (types of brain tumors) in uploaded brain scan images.

⚠️ **Disclaimer:** This is a demonstration project, not a certified medical diagnostic tool. Do not use it to make real medical decisions — always consult a qualified doctor or radiologist.

## How it works

1. Upload a brain scan image (JPG or PNG).
2. The model (`best.pt`, a YOLOv8 detection model trained on two classes: `Astrocytoma` and `Glioblastoma`) runs inference on the image.
3. The app shows a plain-language result:
   - 🔴 **Tumor Found** — with the detected type(s) and confidence score(s), plus a short explanation of what each type means.
   - 🟢 **No Tumor Found** — if no regions pass the confidence threshold.

## Tech stack

- **Streamlit** — web app framework
- **Ultralytics YOLOv8** — object detection model
- **OpenCV (headless)** — image processing
- **Pillow** — image loading

## Project structure

```
├── app.py              # Streamlit app
├── best.pt              # YOLOv8 model weights
├── requirements.txt      # Python dependencies
└── packages.txt          # System-level dependencies (for Streamlit Cloud)
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live demo

Deployed on Streamlit Community Cloud — https://brain-tumor-detector-lbf4pffdaneespdnqnkazg.streamlit.app/



## About the tumor types

- **Astrocytoma** — a tumor that develops from star-shaped glial cells (astrocytes) in the brain or spinal cord. Can range from slow-growing to aggressive depending on grade.
- **Glioblastoma** — the most aggressive type of astrocytoma (grade IV), known for rapid growth and difficulty to treat.

## Notes on accuracy

Model performance depends on the size and quality of the training dataset used. Treat results as illustrative rather than authoritative, and always confirm with a medical professional.
