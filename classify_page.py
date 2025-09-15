import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

st.title("Classification Task")

"""Alege o imagine pe care vrei să o clasifici. Trebuie să aibă una dintre următoarele **extensii**:
\".jpg\", \".jpeg\" sau \".png\". Un dataframe va fi afișat cu rezultatele. 
Rețeaua neuronală a fost făcută și implementată în aplicație folosind tensorflow.
La final apăsați butonul **\"Descărcați CSV\"** pentru a descărca rezultatele in format csv."""

uploaded_files = st.file_uploader(
    "Choose some images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

classification_model = tf.keras.models.load_model(
    "./models/classification/cnn1_model.keras"
)

result_dict = {0: "imagine cos gol", 1: "imagine cos plin"}


@st.cache_data
def convert_for_download(df):
    return df.to_csv().encode("utf-8")


file_number = len(uploaded_files)
if file_number > 0:
    file_names = []
    results = []
    for file in uploaded_files:
        file_names.append(file.name)
        pil_img = Image.open(file)
        resized_image = pil_img.resize((180, 180))
        img_array = tf.keras.utils.img_to_array(resized_image)
        img_array = tf.expand_dims(img_array, 0)
        result = np.argmax(classification_model.predict(img_array))
        results.append(result_dict[result])
    df_dict = {"nume_imagine": file_names, "clasificare": results}
    df = pd.DataFrame(df_dict)
    df
    csv = convert_for_download(df)

    col1, col2 = st.columns([10, 3])

    col2.download_button(
        label="Descărcați CSV",
        data=csv,
        file_name="data.csv",
        mime="text/csv",
        icon=":material/download:",
        type = "primary"
    )
