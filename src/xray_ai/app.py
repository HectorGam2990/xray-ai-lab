from pathlib import Path

import streamlit as st
from PIL import Image

from xray_ai.predict import load_predictor, predict

st.set_page_config(page_title="XRay AI Lab", page_icon="🩻")
st.title("🩻 XRay AI Lab")
st.warning("Proyecto educativo: no diagnostica ni reemplaza a profesionales de la salud.")
checkpoint = st.sidebar.text_input("Checkpoint", "models/best.pt")
uploaded = st.file_uploader("Sube una imagen PNG/JPG", type=["png", "jpg", "jpeg"])


@st.cache_resource
def cached_predictor(path: str, modified: float):
    """modified invalida el caché cuando vuelves a entrenar el checkpoint."""
    del modified
    return load_predictor(path)


if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Imagen recibida", width=500)
    try:
        path = Path(checkpoint)
        model, labels, device = cached_predictor(str(path), path.stat().st_mtime)
        scores = predict(image, model, labels, device)
        for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            st.write(f"**{label}** — {score:.1%}")
            st.progress(score)
        st.caption("Una probabilidad alta no equivale a un diagnóstico ni mide certeza clínica.")
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        st.error(f"No se pudo cargar el modelo: {error}")

