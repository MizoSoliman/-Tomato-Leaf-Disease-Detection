
# ==========================
# 🍅 Tomato Leaf Disease Detection App using YOLOv8
# ==========================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import base64

# =========== Page Setting =============
st.set_page_config(page_title="🍅 Tomato Leaf Disease Detection", layout="centered")

# ======== Background Image (Blurred + Dimmed) ========

def set_background(png_file, brightness=0.35, blur_strength=10):
    with open(png_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                        url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            filter: brightness({brightness});
            backdrop-filter: blur({blur_strength}px);
        }}


        html, body, [class*="st-"] {{
            color: #fdfdfd !important;
            font-weight: 700 !important;
            text-shadow: 0 0 8px rgba(255,255,255,0.6);
        }}


        h1 {{
            text-align: center;
            color: #ffffff !important;
            font-weight: 900 !important;
            text-shadow: 0 0 12px rgba(255,255,255,0.9);
        }}


        h2, h3, h4 {{
            color: #fefefe !important;
            font-weight: 800 !important;
            text-shadow: 0 0 6px rgba(255,255,255,0.7);
        }}


        div.stButton > button:first-child {{
            background-color: #4CAF50;
            color: white;
            font-weight: 700;
            border-radius: 12px;
            border: 1px solid #3e8e41;
            padding: 10px 26px;
            font-size: 18px;
            box-shadow: 0px 0px 10px rgba(72, 239, 128, 0.5);
            transition: 0.3s;
        }}
        div.stButton > button:first-child:hover {{
            background-color: #45a049;
            transform: scale(1.05);
            box-shadow: 0px 0px 15px rgba(72, 239, 128, 0.7);
        }}

        .stAlert {{
            border-radius: 10px;
            font-weight: 700;
            color: #fff !important;
        }}


        hr {{
            border: 1px solid rgba(255,255,255,0.4);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background("181577_web.jpg", brightness=0.85, blur_strength=8)

# =========== Title and Header ============
st.title("🍅 Tomato Leaf Disease Detection")
st.write("Upload **Image** To Detect **Tomato Leaf Diseases** using YOLOv8")

# =========== Loading Model ============
@st.cache_resource
def load_model():
    model_path = "best.pt"
    model = YOLO(model_path)
    return model

model = load_model()

# =========== Upload Section ============
uploaded_file = st.file_uploader("📸 Upload Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Original Image", use_container_width=True)

    if st.button("🔎 Predict Image Disease"):
        with st.spinner("Analyzing.. ⏳"):
            image_path = "temp.jpg"
            image.save(image_path)

            results = model.predict(source=image_path, save=True, conf=0.5)
            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                st.warning("❌ No tomato leaves detected in the image.")
            else:
                result_path = results[0].save_dir + "/" + os.path.basename(image_path)
                st.image(result_path, caption="✅ Detection Result", use_container_width=True)

                st.subheader("📊 Detailed Results:")
                names = model.names
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    st.write(f"• Disease: **{names[cls_id]}** — Confidence: {conf:.2f}")

            st.success("✅ Image analyzed successfully!")

else:
    st.info("Please upload a JPG or PNG image to start prediction.")

# ======= Footer ===================
st.markdown("""
     <hr>
     <p style="text-align:center; color:white; font-weight:800; text-shadow:0 0 8px rgba(255,255,255,0.6);">
     Developed by <b>Mazin Soliman</b> 🌱
     </p>
""", unsafe_allow_html=True)

