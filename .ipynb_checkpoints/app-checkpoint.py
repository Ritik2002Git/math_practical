import cv2
import numpy as np
from PIL import Image, ExifTags
import streamlit as st

# ---------- Add CSS Styling Here ----------
st.markdown(
    """
    <style>
        /* Centered main frame */
        .main {
            max-width: 800px;
            margin: 0 auto;
            padding: 1.5rem;
            background-color: #f9f9f9;
            border-radius: 15px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }

        /* Title style */
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 1rem;
        }

        /* Image size control */
        .stImage > img {
            border-radius: 10px;
            max-height: 400px;
            object-fit: contain;
        }

        /* Sidebar background */
        section[data-testid="stSidebar"] {
            background-color: #f0f0f0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Helper to read camera specs ----
def extract_camera_specs(pil_img):
    try:
        exif = pil_img.getexif()
        if not exif:
            return {}
        exif_dict = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        specs = {
            "Make": exif_dict.get("Make"),
            "Model": exif_dict.get("Model"),
            "LensModel": exif_dict.get("LensModel"),
            "FocalLength": exif_dict.get("FocalLength"),
            "ISOSpeedRatings": exif_dict.get("ISOSpeedRatings"),
            "ExposureTime": exif_dict.get("ExposureTime"),
            "FNumber": exif_dict.get("FNumber"),
        }
        return {k: v for k, v in specs.items() if v is not None}
    except Exception:
        return {}

# ---- Built-in transformation functions ----
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def scale_image(image, scale_factor):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized

def translate_image(image, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted



# ---- Streamlit GUI ----
st.set_page_config(page_title="Image Transformation Tool", layout="wide")

st.markdown('<div class="main">', unsafe_allow_html=True)  # Start styled container

st.title("🖼️ Image Transformation Tool (Library-based)")
st.write("Apply **Rotation**, **Scaling**, or **Translation** using OpenCV built-in functions.")

uploaded_file = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    gray = np.array(image.convert("L"))
    st.image(gray, caption="Original Grayscale Image", use_container_width=True)

    with st.expander("📷 Camera Technical Specs"):
        specs = extract_camera_specs(image)
        if specs:
            st.json(specs)
        else:
            st.write("No EXIF data found.")

    choice = st.selectbox("Choose Transformation", ["Rotate", "Scale", "Translate"])

    if choice == "Rotate":
        angle = st.slider("Rotation Angle (°)", -180, 180, 0)
        rotated = rotate_image(gray, angle)
        st.image(rotated, caption=f"Rotated {angle}°", use_container_width=True)

    elif choice == "Scale":
        scale = st.slider("Scaling Factor", 0.1, 3.0, 1.0)
        scaled = scale_image(gray, scale)
        st.image(scaled, caption=f"Scaled ×{scale}", use_container_width=True)

    elif choice == "Translate":
        tx = st.slider("Translate X (px)", -100, 100, 0)
        ty = st.slider("Translate Y (px)", -100, 100, 0)
        translated = translate_image(gray, tx, ty)
        st.image(translated, caption=f"Translated ({tx}, {ty})", use_column_width=True)

else:
    st.info("👆 Please upload an image to begin.")
