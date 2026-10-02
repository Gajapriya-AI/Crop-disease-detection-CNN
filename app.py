from __future__ import annotations

import io
import sys
from pathlib import Path

import streamlit as st

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, inference


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Tomato Leaf AI",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fff8 0%,
            #ffffff 45%,
            #fff7ed 100%
        );
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero */
    .hero {
        background: linear-gradient(
            135deg,
            #ff512f,
            #f09819
        );
        padding: 38px;
        border-radius: 24px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(255, 81, 47, 0.20);
    }

    .hero h1 {
        color: white;
        font-size: 42px;
        font-weight: 800;
        margin: 0;
    }

    .hero p {
        color: white;
        font-size: 18px;
        margin-top: 10px;
    }

    /* Cards */
    .card {
        background: white;
        border-radius: 18px;
        padding: 22px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.07);
        min-height: 130px;
    }

    .card-title {
        color: #6b7280;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .card-value {
        color: #172554;
        font-size: 24px;
        font-weight: 800;
        margin-top: 10px;
    }

    /* Section headings */
    .section-title {
        color: #172554;
        font-size: 26px;
        font-weight: 800;
        margin-top: 35px;
        margin-bottom: 15px;
    }

    /* Prediction box */
    .prediction-box {
        background: #f0fdf4;
        border: 2px solid #86efac;
        border-radius: 18px;
        padding: 25px;
        margin-top: 20px;
    }

    .disease-box {
        background: #fff7ed;
        border: 2px solid #fdba74;
        border-radius: 18px;
        padding: 25px;
        margin-top: 20px;
    }

    .prediction-title {
        font-size: 28px;
        font-weight: 800;
        color: #172554;
    }

    .confidence {
        font-size: 20px;
        font-weight: 700;
        color: #475569;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 30px 0 10px 0;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading AI model...")
def get_predictor(model_path):
    return inference.DiseasePredictor(str(model_path))


def find_model():
    """
    Use the Custom CNN final model.
    """

    possible_models = [
        config.MODELS_DIR / "crop_disease_best_model.keras",
        config.MODELS_DIR / "cnn_best.keras",
    ]

    for model in possible_models:
        if model.exists():
            return model

    # Fallback: find any keras model
    keras_models = list(config.MODELS_DIR.glob("*.keras"))

    if keras_models:
        return keras_models[0]

    return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🍅 Tomato Leaf AI")

    st.markdown("---")

    st.subheader("🤖 AI Model")

    model_path = find_model()

    if model_path:

        st.success("Model loaded")

        st.write("**Model:** Custom CNN")
        st.write("**Input:** 224 × 224")
        st.write("**Classes:** 5")

    else:

        st.error("No trained model found.")

    st.markdown("---")

    st.subheader("🌿 Disease Classes")

    for class_name in config.CLASSES:
        st.write("•", config.display_label(class_name))

    st.markdown("---")

    st.info(
        "Upload a clear tomato leaf image for prediction."
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🍅 Tomato Leaf AI</h1>
        <p>
            AI-powered tomato leaf disease detection using Deep Learning
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">🚀 AI Model Overview</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Model</div>
            <div class="card-value">Custom CNN</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Input Size</div>
            <div class="card-value">224 × 224</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Classes</div>
            <div class="card-value">5 Classes</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">Dataset</div>
            <div class="card-value">PlantVillage</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📷 Upload Tomato Leaf</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:

    left, right = st.columns([1, 1])

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with left:

        st.subheader("🖼️ Uploaded Image")

        st.image(
            uploaded_file,
            use_container_width=True,
        )

        st.caption(
            f"File: {uploaded_file.name}"
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with right:

        st.subheader("🔍 AI Prediction")

        if model_path is None:

            st.error(
                "Trained model not found in the models folder."
            )

        else:

            predict_clicked = st.button(
                "🔎 Predict Disease",
                type="primary",
                use_container_width=True,
            )

            if predict_clicked:

                with st.spinner(
                    "AI is analyzing the tomato leaf..."
                ):

                    try:

                        predictor = get_predictor(model_path)

                        result = predictor.predict(
                            io.BytesIO(
                                uploaded_file.getvalue()
                            ),
                            filename=uploaded_file.name,
                        )

                        disease = result["disease"]
                        confidence = result["confidence"]
                        confidence_percent = result[
                            "confidence_percent"
                        ]

                        # ------------------------------------------------
                        # RESULT
                        # ------------------------------------------------

                        if result["is_healthy"]:

                            st.success(
                                f"🌿 Healthy Leaf\n\n"
                                f"Confidence: {confidence_percent:.2f}%"
                            )

                        else:

                            st.warning(
                                f"⚠️ Disease Detected: {disease}\n\n"
                                f"Confidence: {confidence_percent:.2f}%"
                            )

                        # Metrics

                        m1, m2 = st.columns(2)

                        with m1:

                            st.metric(
                                "Predicted Class",
                                disease,
                            )

                        with m2:

                            st.metric(
                                "Confidence",
                                f"{confidence_percent:.2f}%",
                            )

                        # Progress

                        st.progress(
                            min(
                                max(
                                    confidence,
                                    0.0,
                                ),
                                1.0,
                            )
                        )

                        # ------------------------------------------------
                        # ALL PROBABILITIES
                        # ------------------------------------------------

                        st.markdown("### 📊 Class Probabilities")

                        ranked = sorted(
                            result["probabilities"].items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )

                        for class_name, probability in ranked:

                            label = config.disease_name(
                                class_name
                            )

                            percentage = probability * 100

                            st.write(
                                f"**{label}** — "
                                f"{percentage:.2f}%"
                            )

                            st.progress(
                                min(
                                    max(
                                        probability,
                                        0.0,
                                    ),
                                    1.0,
                                )
                            )

                        # ------------------------------------------------
                        # LOW CONFIDENCE
                        # ------------------------------------------------

                        if confidence_percent < 60:

                            st.info(
                                "💡 Low confidence prediction. "
                                "Try a clear, well-lit close-up "
                                "image of a single tomato leaf."
                            )

                    except Exception as error:

                        st.error(
                            f"Prediction failed: {error}"
                        )


# ============================================================
# INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">🌱 Supported Diseases</div>',
    unsafe_allow_html=True,
)

disease_cols = st.columns(5)

for index, class_name in enumerate(config.CLASSES):

    with disease_cols[index]:

        st.info(
            config.display_label(class_name)
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    🍅 <b>Tomato Leaf AI</b><br>

    Built with TensorFlow / Keras + Streamlit<br>

    Dataset: PlantVillage — 5 Tomato Leaf Classes<br>

    <br>

    For research and educational purposes only.

    </div>
    """,
    unsafe_allow_html=True,
)