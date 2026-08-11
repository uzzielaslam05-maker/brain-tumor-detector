import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="centered")

# ---------- Custom styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0D0B1A;
    background-image: radial-gradient(rgba(157, 127, 234, 0.15) 1px, transparent 1px);
    background-size: 22px 22px;
    color: #EDEAFB;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
footer { visibility: hidden; }

.bt-header {
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(157, 127, 234, 0.25);
    margin-bottom: 8px;
}
.bt-title {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 2.05rem;
    background: linear-gradient(90deg, #C9B6FA, #E85D9E);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.01em;
}
.bt-sub {
    color: #8F87B0;
    font-size: 0.95rem;
    margin-top: 8px;
}
.bt-warn {
    font-size: 0.8rem;
    color: #E8B4D8;
    background: rgba(232, 93, 158, 0.08);
    border-left: 2px solid #E85D9E;
    padding: 10px 14px;
    margin: 18px 0 26px 0;
    line-height: 1.5;
    border-radius: 0 6px 6px 0;
}
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(157, 127, 234, 0.4);
    border-radius: 10px;
    background: rgba(157, 127, 234, 0.04);
    padding: 6px;
}
.bt-result-card {
    border-radius: 14px;
    padding: 22px 24px;
    margin-top: 20px;
    border: 1px solid rgba(157, 127, 234, 0.25);
}
.bt-result-found {
    background: linear-gradient(135deg, rgba(232, 93, 158, 0.12), rgba(157, 127, 234, 0.08));
}
.bt-result-clear {
    background: rgba(157, 127, 234, 0.05);
}
.bt-badge {
    display: inline-block;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    font-size: 1.1rem;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 14px;
}
.bt-badge-found {
    background: rgba(232, 93, 158, 0.18);
    color: #F3A6CC;
    border: 1px solid rgba(232, 93, 158, 0.4);
}
.bt-badge-clear {
    background: rgba(143, 217, 170, 0.14);
    color: #8FD9AA;
    border: 1px solid rgba(143, 217, 170, 0.4);
}
.bt-detection-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.92rem;
    padding: 8px 0;
    border-bottom: 1px solid rgba(237, 234, 251, 0.08);
}
.bt-detection-row:last-child { border-bottom: none; }
.bt-conf { color: #C9B6FA; font-weight: 600; }
.bt-note {
    font-size: 0.82rem;
    color: #8F87B0;
    margin-top: 14px;
    line-height: 1.6;
    border-top: 1px solid rgba(237, 234, 251, 0.08);
    padding-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="bt-header">
    <div class="bt-title">Brain Tumor Detector</div>
    <div class="bt-sub">Upload a brain scan to check for signs of Astrocytoma or Glioblastoma.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="bt-warn">
⚠ This is a demonstration model, not a medical diagnostic tool. Do not use it to make real medical decisions — consult a qualified doctor.
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload a brain scan image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

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
        st.markdown('<div class="bt-result-card bt-result-found">', unsafe_allow_html=True)
        st.markdown('<span class="bt-badge bt-badge-found">🔴 Tumor Found</span>', unsafe_allow_html=True)
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.markdown(
                f'<div class="bt-detection-row"><span>{label}</span>'
                f'<span class="bt-conf">{conf:.2%}</span></div>',
                unsafe_allow_html=True
            )
        st.markdown("""
        <div class="bt-note">
        Astrocytoma and Glioblastoma are both types of brain tumors that originate from glial (supportive)
        cells in the brain. Glioblastoma is generally more aggressive than Astrocytoma. This tool does not
        confirm a diagnosis — only a doctor can do that.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="bt-result-card bt-result-clear">', unsafe_allow_html=True)
        st.markdown('<span class="bt-badge bt-badge-clear">🟢 No Tumor Found</span>', unsafe_allow_html=True)
        st.markdown('<div class="bt-note">The model did not detect any signs of a tumor in this image.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
