"""Streamlit web app: upload a tomato-leaf image and get a disease prediction.

Run from the project root:

    Windows (VS Code terminal):   py -m streamlit run app.py
    macOS / Linux:                python3 -m streamlit run app.py

The browser opens automatically at http://localhost:8501.  The app loads the
best MobileNetV3 model (``models/mobilenet_v3_best.keras``) by default; the
CNN baseline can be selected in the sidebar if it has been trained too.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Make "from src import ..." work no matter how Streamlit launches this file.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src import config, inference  # noqa: E402


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Tomato Leaf Disease Detection",
    page_icon="🍅",
    layout="centered",
)

ALLOWED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]


@st.cache_resource(show_spinner="Loading model, please wait ...")
def get_predictor(model_path_str: str) -> inference.DiseasePredictor:
    """Load a model once and reuse it for every prediction (cached)."""
    return inference.DiseasePredictor(model_path_str)


def find_available_models() -> dict[str, Path]:
    """Return {display name: path} for every trained model found in models/."""
    available: dict[str, Path] = {}
    if config.MOBILENET_MODEL_PATH.is_file():
        available["MobileNetV3 (recommended)"] = config.MOBILENET_MODEL_PATH
    if config.CNN_MODEL_PATH.is_file():
        available["Custom CNN baseline"] = config.CNN_MODEL_PATH
    # Any other .keras files the user may have saved in models/.
    for path in sorted(config.MODELS_DIR.glob("*.keras")):
        if path not in available.values():
            available[path.stem] = path
    return available


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🍅 Settings")

    available_models = find_available_models()
    if available_models:
        model_label = st.selectbox(
            "Choose a trained model",
            options=list(available_models),
            index=0,
            help="Train models with 'py src\\train_mobilenet.py' and 'py src\\train_cnn.py'.",
        )
        model_path = available_models[model_label]
        st.caption(f"Model file: `{model_path.name}`")
    else:
        model_path = None
        st.error(
            "No trained model found in the `models/` folder.\n\n"
            "Train one first from the project root:\n\n"
            "`py src\\train_mobilenet.py`"
        )

    st.divider()
    st.subheader("About this app")
    st.markdown(
        "This app detects diseases in **tomato leaves** using a MobileNetV3 "
        "convolutional neural network trained on the PlantVillage dataset.\n\n"
        "**Classes it knows:**"
    )
    for class_name in config.CLASSES:
        st.markdown(f"- {config.display_label(class_name)}")
    st.caption(
        f"Images are resized to {config.IMG_SIZE}x{config.IMG_SIZE} pixels "
        "before prediction. Upload a clear, close-up photo of a single leaf "
        "for the best results."
    )

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("🍅 Tomato Leaf Disease Detection")
st.markdown(
    "Upload a photo of a tomato leaf and the model will predict whether it is "
    "**healthy** or affected by one of **four diseases**, together with a "
    "confidence percentage."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image (JPG, JPEG, PNG, BMP or WEBP)",
    type=ALLOWED_UPLOAD_TYPES,
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info("👆 No image uploaded yet. Select an image file to get started.")
    st.stop()

# --- Show the uploaded image ------------------------------------------------
left_column, right_column = st.columns([1, 1])
with left_column:
    st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}")

with right_column:
    st.subheader("Prediction")

    # --- Validate the file extension before doing any work -----------------
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix.lstrip(".") not in ALLOWED_UPLOAD_TYPES:
        st.error(
            f"Unsupported file type '{suffix}'. "
            f"Please upload one of: {', '.join(ALLOWED_UPLOAD_TYPES)}."
        )
        st.stop()

    if model_path is None:
        st.error(
            "No trained model is available yet. See the sidebar for the "
            "training command, then refresh this page."
        )
        st.stop()

    predict_clicked = st.button("🔎 Predict disease", type="primary")

    if predict_clicked:
        try:
            predictor = get_predictor(str(model_path))
            # Wrap the uploaded bytes in a file-like object for the predictor.
            result = predictor.predict(
                io.BytesIO(uploaded_file.getvalue()),
                filename=uploaded_file.name,
            )
        except (FileNotFoundError, ValueError, ImportError) as error:
            st.error(f"Prediction failed: {error}")
            st.stop()
        except Exception as error:  # noqa: BLE001 - show any unexpected problem
            st.error(f"Unexpected error during prediction: {error}")
            st.exception(error)
            st.stop()

        # --- Show the result ------------------------------------------------
        confidence_percent = result["confidence_percent"]

        if result["is_healthy"]:
            st.success(f"✅ This leaf looks **Healthy**! ({confidence_percent:.1f}% confidence)")
        else:
            st.warning(
                f"⚠️ Disease detected: **{result['disease']}** "
                f"({confidence_percent:.1f}% confidence)"
            )

        st.metric(
            label=f"Confidence for '{result['disease']}'",
            value=f"{confidence_percent:.1f}%",
        )
        st.progress(min(max(result["confidence"], 0.0), 1.0))

        # Probabilities of every class, highest first.
        with st.expander("Probabilities for all classes", expanded=True):
            ranked = sorted(
                result["probabilities"].items(), key=lambda item: item[1], reverse=True
            )
            for class_name, probability in ranked:
                label = config.disease_name(class_name)
                st.write(f"**{label}** - {probability * 100:.2f}%")
                st.progress(min(max(probability, 0.0), 1.0))

        if confidence_percent < 60.0:
            st.info(
                "ℹ️ The confidence is low. Try a sharper, well-lit close-up "
                "photo of a single leaf."
            )

st.divider()
st.caption(
    "Built with TensorFlow/Keras + Streamlit. Dataset: PlantVillage "
    "(5 tomato classes). For research/education - not a substitute for "
    "expert agricultural advice."
)
