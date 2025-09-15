import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import zipfile

st.title("Detection task")

"""Alegeți o imagine pe care vrei să faceți detecție de obiecte."""

conf_value = st.slider("Încrederea predicției", min_value=0.0, max_value=1.0, value=0.7)
iou_value = st.slider("iou", min_value=0.0, max_value=1.0, value=0.7)

uploaded_files = st.file_uploader(
    "Choose some images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

model = YOLO("./models/detection/best.pt")

file_number = len(uploaded_files)


def predict_image(img, conf_threshold, iou_threshold):
    """Predicts objects in an image using a YOLO11 model with adjustable confidence and IOU thresholds."""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=180,
    )

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    return im


if file_number > 0:
    resulted_images = []
    file_names = []
    for file in uploaded_files:
        file_names.append(file.name)
        pil_img = Image.open(file)
        result = predict_image(
            pil_img, conf_threshold=conf_value, iou_threshold=iou_value
        )
        resulted_images.append(result)

    # Create a zip in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, img in enumerate(resulted_images):
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            zip_file.writestr(f"{file_names[idx]}", img_bytes.read())
    zip_buffer.seek(0)

    col1, col2 = st.columns([3, 1])

    # Download button
    col2.download_button(
        label="Descarcă imaginile",
        data=zip_buffer,
        file_name="images.zip",
        mime="application/zip",
    )

    for img in resulted_images:
        st.image(img)
