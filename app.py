import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

from inference import predict

st.set_page_config(
    page_title="Satellite Land Cover Classification",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satellite Land Cover Classification")

st.write(
    """
Upload a satellite image and the Swin Transformer model
will predict its land-cover category.
"""
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    predictions = predict(image)

    with col2:

        best = predictions[0]

        st.success(
            f"Prediction: **{best['label']}**"
        )

        st.metric(
            "Confidence",
            f"{best['confidence']*100:.2f}%"
        )

        st.subheader("Top 5 Predictions")

        labels = []
        scores = []

        for item in predictions:

            labels.append(item["label"])

            scores.append(item["confidence"] * 100)

            st.write(
                f"{item['label']} : {item['confidence']*100:.2f}%"
            )

        fig, ax = plt.subplots(figsize=(7,4))

        ax.bar(labels, scores)

        ax.set_ylabel("Confidence (%)")

        ax.set_ylim(0,100)

        plt.xticks(rotation=20)

        st.pyplot(fig)