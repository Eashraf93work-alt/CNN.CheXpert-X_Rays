"""Streamlit application for CheXpert chest X-ray classification.

Provides a clean, professional, and responsive web interface for uploading
chest X-ray images, preprocessing them, running inference using EfficientNetB0,
and displaying the multi-label pathology classification results.
"""

import os
import time
import logging
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Import local utility functions
from utils import preprocess_image, build_model, predict, CHEXPERT_LABELS

# Configure logger
logger = logging.getLogger("CheXpertApp")

# Setup Page Configuration
st.set_page_config(
    page_title="CheXpert X-ray Diagnostics Portal",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* Apply fonts */
        .stApp {
            font-family: 'Inter', sans-serif;
        }
        
        /* Style headers */
        h1, h2, h3 {
            color: #1e293b;
        }
        
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.2rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .subtitle {
            font-size: 1.1rem;
            color: #64748b;
            margin-bottom: 2rem;
        }
        
        /* Metric cards custom design */
        .metric-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: all 0.2s ease-in-out;
        }
        .metric-card:hover {
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        
        .metric-card-high {
            background-color: #fef2f2;
            border: 1px solid #fee2e2;
            border-left: 5px solid #ef4444;
        }
        
        .metric-card-normal {
            background-color: #f0fdf4;
            border: 1px solid #dcfce7;
            border-left: 5px solid #22c55e;
        }
        
        .metric-label {
            font-size: 0.95rem;
            font-weight: 600;
            color: #334155;
        }
        
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 0.2rem;
        }
        
        /* Status Badges */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            font-size: 0.8rem;
            font-weight: 600;
            border-radius: 50px;
            text-align: center;
        }
        .badge-danger {
            background-color: #fee2e2;
            color: #991b1b;
        }
        .badge-success {
            background-color: #dcfce7;
            color: #166534;
        }
        .badge-neutral {
            background-color: #f1f5f9;
            color: #475569;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_cached_model(weights_path: str):
    """Load and cache the CheXpert classification model using Streamlit cache_resource.

    Args:
        weights_path (str): File path to model weights .h5 file.

    Returns:
        tf.keras.Model: Loaded Keras model instance.
    """
    logger.info(f"App-level caching triggered for model weights: {weights_path}")
    try:
        # Build model and load weights
        model = build_model(num_classes=len(CHEXPERT_LABELS), weights_path=weights_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model in cached resource: {e}", exc_info=True)
        raise e


def display_results_chart(results: dict, threshold: float):
    """Display predictions in a styled horizontal Matplotlib bar chart.

    Args:
        results (dict): Dictionary of predictions.
        threshold (float): Classification threshold.
    """
    # Sort items for plotting
    sorted_items = sorted(results.items(), key=lambda item: item[1])
    labels = [x[0] for x in sorted_items]
    probs = [x[1] for x in sorted_items]

    # Style colors based on whether they exceed the threshold
    colors = ["#ef4444" if p >= threshold else "#3b82f6" for p in probs]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    bars = ax.barh(labels, probs, color=colors, height=0.6, edgecolor='none')

    # Add threshold line
    ax.axvline(x=threshold, color="#f59e0b", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.2f})")

    # Grid & styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.xaxis.grid(True, linestyle=':', alpha=0.6, color='#cbd5e1')
    ax.set_axisbelow(True)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Probability", fontsize=10, fontweight='bold', color='#475569')
    ax.tick_params(colors='#475569', labelsize=9)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor='#cbd5e1')

    # Add numeric labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1%}",
            ha='left',
            va='center',
            fontsize=8.5,
            color='#1e293b',
            fontweight='semibold'
        )

    plt.tight_layout()
    st.pyplot(fig)


def main():
    """Main application loop."""
    logger.info("Initializing CheXpert Streamlit UI dashboard")

    # Sidebar Panel
    st.sidebar.markdown("### 🎛️ Diagnostic Configuration")
    
    # Weights configuration
    default_weights_path = "./feature.weights.h5"
    weights_path = st.sidebar.text_input(
        "Model Weights Path (.h5)",
        value=default_weights_path,
        help="Specify the path to the trained h5 model weights file."
    )

    # Classification Threshold slider
    threshold = st.sidebar.slider(
        "Pathology Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Probabilities greater than or equal to this threshold are flagged as positive findings."
    )

    # Metadata & environment info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧬 Diagnostic Specifications")
    st.sidebar.info(
        "**Core Backbone:** EfficientNetB0\n"
        "**Target Dataset:** CheXpert\n"
        "**Preprocessed Input:** 224x224x3\n"
        "**Inference:** Multi-label Sigmoid"
    )

    # Main dashboard header
    st.markdown('<div class="main-title">🩻 CheXpert Chest X-ray Diagnostic Portal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Production-grade AI-assisted diagnostics tool powered by EfficientNetB0</div>',
        unsafe_allow_html=True
    )

    # Validate weights path exists
    if not os.path.exists(weights_path):
        st.error(f"⚠️ Model weights file not found at path: `{weights_path}`. Please verify the path in the sidebar.")
        st.stop()

    # Load Model (Cached)
    model = None
    with st.spinner("⏳ Loading medical model backbone... This may take a moment on first execution."):
        try:
            model = load_cached_model(weights_path)
            logger.info("CheXpert model loaded successfully inside app.py UI context.")
        except Exception as e:
            st.error(f"❌ Failed to construct or load the model. Error details: {e}")
            logger.critical(f"Failed model load: {e}", exc_info=True)
            st.stop()

    # Layout: 2 Columns for Uploading and Predictions
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### 📁 Upload Patient Radiograph")
        uploaded_file = st.file_uploader(
            "Select a Chest X-ray image (JPG, JPEG, or PNG)",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPEG, PNG. Grayscale radiographs are automatically converted to RGB."
        )

        if uploaded_file is not None:
            # Display uploaded image preview
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Chest X-ray Radiograph", use_container_width=True)
            except Exception as load_err:
                st.error(f"❌ Critical Error: Could not load the uploaded image. Details: {load_err}")
                logger.error(f"Failed to open uploaded image file: {load_err}", exc_info=True)
                st.stop()
        else:
            # Welcome/Instructions display when no image is uploaded
            st.info("💡 Please upload a patient's front-facing or lateral chest X-ray to initiate pathology assessment.")
            
            # Show a brief warning about AI diagnostics usage
            st.warning(
                "⚠️ **Disclaimer:** This tool is for clinical research and educational demonstration purposes only. "
                "It is not FDA approved and should not be used as a replacement for professional clinical judgment."
            )

    with col_right:
        st.markdown("### 📊 Diagnostic Findings")

        if uploaded_file is not None:
            # Run inference
            results = None
            preprocess_err = False
            inference_err = False

            with st.spinner("🔄 Preprocessing radiograph & executing neural network..."):
                # Preprocess image
                try:
                    preprocessed_img = preprocess_image(uploaded_file)
                except Exception as prep_ex:
                    preprocess_err = True
                    st.error(f"❌ Preprocessing Error: Failed to scale and format image. Details: {prep_ex}")
                    logger.error(f"Preprocessing exception in UI: {prep_ex}", exc_info=True)

                # Predict
                if not preprocess_err:
                    try:
                        results = predict(model, preprocessed_img)
                    except Exception as inf_ex:
                        inference_err = True
                        st.error(f"❌ Inference Error: Failed running the neural network forward pass. Details: {inf_ex}")
                        logger.error(f"Inference exception in UI: {inf_ex}", exc_info=True)

            if results is not None:
                # Execution Success
                st.success("✅ Assessment complete. Review predictions below.")

                # List detected pathologies
                detected = [name for name, prob in results.items() if prob >= threshold]
                
                if detected:
                    st.markdown("#### 🚨 Flagged Findings")
                    for d in detected:
                        prob_val = results[d]
                        st.markdown(
                            f'<span class="badge badge-danger">Detected</span> **{d}** (Probability: `{prob_val:.1%}`)',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown("#### 🚨 Flagged Findings")
                    st.markdown(
                        f'<span class="badge badge-success">Clear</span> No pathologies detected above the threshold of `{threshold:.1%}`.',
                        unsafe_allow_html=True
                    )

                # Tabs for different visualization perspectives
                tab_chart, tab_metrics = st.tabs(["📉 Probability Visualizer", "📋 Detailed Scores"])

                with tab_chart:
                    # Render styled Matplotlib chart
                    display_results_chart(results, threshold)

                with tab_metrics:
                    # Display metrics in standard layout
                    st.markdown("#### Pathology Scorecard")
                    metric_cols = st.columns(2)
                    
                    for i, (pathology, prob) in enumerate(results.items()):
                        target_col = metric_cols[i % 2]
                        is_flagged = prob >= threshold
                        card_class = "metric-card-high" if is_flagged else "metric-card-normal"
                        badge_html = (
                            '<span class="badge badge-danger">Exceeds Threshold</span>' 
                            if is_flagged else '<span class="badge badge-neutral">Normal</span>'
                        )
                        
                        target_col.markdown(f"""
                            <div class="metric-card {card_class}">
                                <div class="metric-label">{pathology}</div>
                                <div class="metric-value">{prob:.1%}</div>
                                <div style="margin-top: 5px;">{badge_html}</div>
                            </div>
                            <br>
                        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
