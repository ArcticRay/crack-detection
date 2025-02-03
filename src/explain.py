import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# LIME & scikit-image
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model


# Configuration
MODEL_PATH = 'models/best_model.keras'  # trained grayscale image
IMG_PATH = 'data/raw/Positive/00023.jpg'  # test image
IMG_SIZE = (128, 128)
CLASS_INDEX = 1  # 0=Negative, 1=Positive 

OUTPUT_DIR = 'outputs/lime'  
OUTPUT_FILENAME = 'lime_explanation.png'  # file name for the saved figure
os.makedirs(OUTPUT_DIR, exist_ok=True)    # make sure directory exists


# LIME-Helper
def model_predict(images: list) -> np.ndarray:
    """
    LIME calls this function repeatedly to obtain predictions for perturbated images.
    
    images: List of NumPy arrays; 
            For grayscale images, LIME may generate isolated (H,W,3)...
            We force everything here to (H,W,1).

    Return: 2-column array (batch_size, 2) => [p(neg), p(pos)] for each instance.
    """
    x_batch = []
    for img in images:
        # if shape == (H, W), expand to (H,W,1)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)  # (H,W)->(H,W,1)

        # if shape == (H, W, 3), reduce to gray
        if img.shape[-1] == 3:
            # Averaged gray values or z. B. OpenCV-Conversion
            # Here: Average => shape (H, W)
            gray = np.mean(img, axis=-1, keepdims=True)  # (H,W,1)
            img = gray

        x_batch.append(img)

    x_batch = np.array(x_batch)  # => shape (batch,128,128,1)
    # Prediction => p(pos)
    preds_pos = model.predict(x_batch)  # => shape (batch,1)
    
    # From this p(neg)=1-p(pos)
    preds_neg = 1.0 - preds_pos
    # => 2 Columns: [p(neg), p(pos)]
    return np.hstack([preds_neg, preds_pos])  # => shape (batch,2)


if __name__ == "__main__":
    # Load Model
    model = load_model(MODEL_PATH)
    model.summary()  # For control

    # load test image
    if not os.path.isfile(IMG_PATH):
        raise FileNotFoundError(f"Image not fdound: {IMG_PATH}")

    raw_img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        raise FileNotFoundError(f"Could not load {IMG_PATH}!")
    # Resize to (128,128)
    raw_img = cv2.resize(raw_img, IMG_SIZE).astype('float32')

    # Normalize 0..1
    raw_img /= 255.0

    # LIME-Explainer
    explainer = lime_image.LimeImageExplainer()

    # Quickshift without Lab => Grayscale-friendly
    segmentation_fn = SegmentationAlgorithm(
        'quickshift',
        kernel_size=4,
        max_dist=200,
        ratio=0.2,
        convert2lab=False  
    )

    # calculate LIME
    explanation = explainer.explain_instance(
        image=raw_img,  # shape=(128,128) => grayscale
        classifier_fn=model_predict,
        segmentation_fn=segmentation_fn,
        top_labels=2,     # 0=Neg,1=Pos
        num_samples=1000  # for example 1000 Perturbations
    )

    # Visualize
    from_label = CLASS_INDEX  # 1 => Positive
    lime_img, mask = explanation.get_image_and_mask(
        label=from_label,
        positive_only=True, 
        hide_rest=False,
        num_features=5,   # 5 importatn Superpixel
        min_weight=0.0
    )

    # Plot
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("Original Gray")
    plt.imshow(raw_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("LIME Explanation")
    # mark_boundaries => Draws segmented boundaries
    # lime_img could be float => Convert to uint8
    plt.imshow(mark_boundaries(lime_img.astype('uint8'), mask))
    plt.axis('off')

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=150)
    print(f"LIME explanation saved to: {output_path}")

    plt.show()
