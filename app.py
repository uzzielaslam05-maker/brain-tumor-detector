import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detector", layout="centered")
st.title("Brain Tumor Detection")
st.write("Upload a brain scan image and the model will check for signs of a tumor.")
st.caption("⚠️ This is a demonstration model, not a medical diagnostic tool. Do not use it to make real medical decisions — consult a qualified doctor.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Choose a brain scan image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running detection..."):
        results = model.predict(image, conf=0.10)
        annotated = results[0].plot()

    annotated_rgb = annotated[:, :, ::-1]
    st.image(annotated_rgb, caption="Detections", use_container_width=True)

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        st.subheader("🔴 Tumor Found")
        st.write("The model detected region(s) that may indicate a tumor:")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.write(f"- Type: **{label}** — Confidence: {conf:.2%}")
        st.caption(
            "Astrocytoma and Glioblastoma are both types of brain tumors that originate "
            "from glial (supportive) cells in the brain. Glioblastoma is generally more "
            "aggressive than Astrocytoma. This tool does not confirm a diagnosis — only a "
            "doctor can do that."
        )
    else:
        st.subheader("🟢 No Tumor Found")
        st.write("The model did not detect any signs of a tumor in this image.")
