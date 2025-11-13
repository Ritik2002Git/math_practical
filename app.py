import cv2
import numpy as np
from PIL import Image
import streamlit as st
import cv2
import numpy as np

GREEN = (0, 255, 0)

def resize_with_aspect_ratio(image, max_dim=512):  #resize image
    
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h))
    return image

def save_gray_as_matrix(gray_image):  # this convert into matrix
    
    matrix = np.array(gray_image, dtype=np.uint8)
    np.save("grayscale_matrix.npy", matrix)
    return matrix

def rotate_image(img, angle):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

def translate_image(img, tx, ty):
    # keep same canvas, pad with white so the shift is obvious
    h, w = img.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

def scale_image(img, scale):
    
    h, w = img.shape
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # white canvas same size as original
    canvas = np.full((h, w), 0, dtype=img.dtype)

    # top-left where the resized image will be placed (centered)
    y0 = max(0, (h - new_h) // 2)
    x0 = max(0, (w - new_w) // 2)

    # paste with clipping if scaled image is larger than canvas
    y1 = min(h, y0 + new_h)
    x1 = min(w, x0 + new_w)
    crop_h = y1 - y0
    crop_w = x1 - x0

    canvas[y0:y1, x0:x1] = resized[:crop_h, :crop_w]
    return canvas



# used StreamLit for python web app
st.set_page_config(page_title="Image Transformation Tool", layout="wide")

st.title(" Image Transformation Tool Practical ")
st.write("Apply **Rotation**, **Scaling**, or **Translation** .")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file:

    image = Image.open(uploaded_file)

    resize_image = resize_with_aspect_ratio(image, max_dim=512)
    gray = np.array(resize_image.convert("L"))
    image_array = np.array(image)
    h, w = image_array.shape[:2]
    new_h, new_w = gray.shape[:2]

    # Used nested columns
    # two columns are used for display image matrixand original and gray image
    col1, col2 = st.columns(2, gap="large" )

    with col1:
        matrix = save_gray_as_matrix(gray)
        st.write("Image matrix:", gray[:10, :10])

    
    # this coloum for image original and gray
    with col2:
        nested_col1, nested_col2 = st.columns(2)

        with nested_col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=False)
            st.write("Orignal Image", h, "x", w)


        with nested_col2:
            st.subheader("Grayscale Image")
            st.image(gray, use_container_width=False)
            st.write("Resized Image", new_h, "x", new_w)


    # used radio button for Transformation control
    st.subheader("Image Transformation Controls")

    radio_button = st.radio(
        "select tranformation type",
        ["Rotate", "Scale", "Translate"],
        horizontal=True
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        
            if radio_button == "Rotate":
                # use slide for taking user input
                angle = st.slider("Rotation Angle (°)", -180, 180, 0)
                transformed = rotate_image(gray, angle)
                caption = f"Rotated {angle}°"

            elif radio_button == "Scale":
                # use slide for taking user input
                scale = st.slider("Scaling Factor", 0.1, 3.0, 1.0)
                transformed = scale_image(gray, scale)
                caption = f"Scaled ×{scale}"

            elif radio_button == "Translate":
                # use slide for taking user input
                tx = st.slider("Translate X (px)", -100, 100, 0)
                ty = st.slider("Translate Y (px)", -100, 100, 0)
                transformed = translate_image(gray, tx, ty)
                caption = f"Translated ({tx}, {ty})"

    with col_right:
        st.subheader("Transformed Image")
        st.image(transformed, caption=caption, use_container_width=512)

else:
    st.info("Please upload an image to begin.")
    
