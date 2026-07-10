"""Utility functions for CheXpert medical image classification.

This module contains functions for image loading, preprocessing, model
construction, loading pre-trained weights, and running inference.

All code follows PEP 8 standards and uses standard Python logging.
"""

import time
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CheXpertUtils")

# CheXpert Pathology labels (14 outputs)
CHEXPERT_LABELS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]


def preprocess_image(image_input) -> np.ndarray:
    """Preprocess a chest X-ray image for EfficientNetB0 inference.

    Args:
        image_input: Can be a file path (str), a PIL Image object,
                     or raw bytes.

    Returns:
        np.ndarray: Preprocessed image batch of shape (1, 224, 224, 3)
                    with pixel values in the range [0.0, 255.0].

    Raises:
        ValueError: If the image cannot be loaded or processed.
    """
    start_time = time.time()
    try:
        if isinstance(image_input, str):
            logger.info(f"Loading image from file path: {image_input}")
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            logger.info("Processing PIL Image object")
            img = image_input
        else:
            # Assume raw bytes or file-like object
            logger.info("Loading image from raw bytes/file-like object")
            img = Image.open(image_input)

        # Log original metadata
        logger.info(f"Original image format: {img.format}, size: {img.size}, mode: {img.mode}")

        # Convert to RGB (EfficientNetB0 expects 3 channels)
        if img.mode != "RGB":
            logger.info(f"Converting image mode from {img.mode} to RGB")
            img = img.convert("RGB")

        # Resize to 224x224 (Bilinear interpolation is used in tf.image.resize)
        logger.info("Resizing image to 224x224")
        img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)

        # Convert to numpy array of float32
        img_array = np.array(img_resized, dtype=np.float32)

        # Expand dims to represent a batch of size 1: (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)

        elapsed_time = time.time() - start_time
        logger.info(f"Image preprocessing successful. Shape: {img_batch.shape}, time taken: {elapsed_time:.4f}s")
        return img_batch

    except Exception as e:
        logger.error(f"Error during image preprocessing: {e}", exc_info=True)
        raise ValueError(f"Failed to preprocess image: {e}")


def build_model(num_classes: int = 14, weights_path: str = None) -> models.Model:
    """Build the EfficientNetB0 classification model and load custom weights.

    Args:
        num_classes (int): Number of target output classes (default: 14).
        weights_path (str): Optional path to custom weights (.h5 file).

    Returns:
        models.Model: Compiled Keras model instance.
    """
    logger.info("Building CheXpert model architecture...")

    # Data augmentation sequence (exact replica of training architecture to match weights topology)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.15)
    ], name="sequential")

    # EfficientNetB0 base backbone without top classification layers
    base_model = tf.keras.applications.EfficientNetB0(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    if weights_path:
        logger.info(f"Loading model weights from: {weights_path}")
        start_time = time.time()
        try:
            # First attempt: load exact topology weights
            model.load_weights(weights_path)
            logger.info("Model weights loaded successfully using exact topology.")
        except Exception as e:
            logger.warning(
                f"Direct topology weight loading failed: {e}. "
                "Attempting loading by name & skipping mismatches as fallback."
            )
            try:
                model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                logger.info("Model weights loaded successfully using by_name=True fallback.")
            except Exception as fallback_err:
                logger.error(f"Fallback weight loading also failed: {fallback_err}", exc_info=True)
                raise fallback_err
        elapsed_time = time.time() - start_time
        logger.info(f"Model weight loading time taken: {elapsed_time:.4f}s")

    return model


def predict(model: models.Model, preprocessed_image: np.ndarray) -> dict:
    """Run model inference on a preprocessed chest X-ray image batch.

    Args:
        model (models.Model): Loaded and compiled model.
        preprocessed_image (np.ndarray): Image array of shape (1, 224, 224, 3).

    Returns:
        dict: A dictionary mapping pathology names to prediction probabilities.
    """
    logger.info("Starting model inference...")
    start_time = time.time()

    try:
        # Run model forward pass. Calling the model directly (training=False)
        # is faster than model.predict() for single images since it avoids overhead.
        predictions = model(preprocessed_image, training=False)
        probabilities = predictions.numpy()[0]

        elapsed_time = time.time() - start_time
        logger.info(f"Inference completed in {elapsed_time:.4f}s")

        # Map predictions to class labels
        results = {}
        for label, prob in zip(CHEXPERT_LABELS, probabilities):
            results[label] = float(prob)
            logger.debug(f"Pathology: {label:<30} Probability: {prob:.4f}")

        return results

    except Exception as e:
        logger.error(f"Error during model inference: {e}", exc_info=True)
        raise RuntimeError(f"Inference failed: {e}")
