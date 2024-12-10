import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd

def preprocess_image(img_path, image_size=(128, 128)):
    """
    Loads and preprocesses a single image

    Args:
        img_path (str): Path to image
        image_size (tuple): wanted image size (width, height)

    Returns:
        np.ndarray: Preprocessd image as NumPy-Array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image could not be load: {img_path}")
    img = cv2.resize(img, image_size)
    img = img.astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)  # (128, 128, 1)
    img = np.expand_dims(img, axis=0)   # (1, 128, 128, 1)
    return img

def predict_image(model, img_array):
    """
    Generates a prediction for a single preprocessed image

    Args:
        model (tensorflow.keras.models.Model): loaded model.
        img_array (np.ndarray): preprocessed image.

    Returns:
        float: Probability for positive class
        int: class prediction (0=Negative, 1=Positive).
    """
    pred_prob = model.predict(img_array)[0][0]
    pred_class = int(pred_prob > 0.5)
    return pred_prob, pred_class

def plot_image_with_prediction(img_path, pred_class, pred_prob):
    """
    Plots image with prediction

    Args:
        img_path (str): Path to image
        pred_class (int): Prediction
        pred_prob (float): probability for the prediction
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    plt.imshow(img, cmap='gray')
    title = f"Prediction: {'Positive' if pred_class == 1 else 'Negative'} ({pred_prob:.2f})"
    plt.title(title)
    plt.axis('off')
    plt.show()

def main(args):
    """
    Main Function

    Args:
        args (Namespace): Command Line Arguments.

    Returns:
        None
    """
    # Check if directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # Load trained model
    model_path = 'models/best_model.keras'
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = load_model(model_path)
    print("Model load successfulyy.")

    # Check if its an image or a directory
    if os.path.isfile(args.image_path):
        image_paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        # COllect all images in directory
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        image_paths = [os.path.join(args.image_path, fname) for fname in os.listdir(args.image_path)
                       if fname.lower().endswith(valid_extensions)]
        if not image_paths:
            raise ValueError(f"Keine gültigen Bilddateien im Verzeichnis gefunden: {args.image_path}")
    else:
        raise ValueError(f"Ungültiger Bildpfad: {args.image_path}")

    # List to save results
    results = []

    for img_path in image_paths:
        try:
            # image preprocessing
            img_array = preprocess_image(img_path)
            
            # generate prediction
            pred_prob, pred_class = predict_image(model, img_array)
            
            # save result
            results.append({
                'Image': os.path.basename(img_path),
                'Prediction': 'Positive' if pred_class == 1 else 'Negative',
                'Probability': pred_prob
            })
            
            # Optional: Show image with prediction
            if args.show:
                plot_image_with_prediction(img_path, pred_class, pred_prob)
            
            print(f"{os.path.basename(img_path)}: {results[-1]['Prediction']} ({pred_prob:.2f})")
        
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {img_path}: {e}")

    # save results
    if results:
        df = pd.DataFrame(results)
        output_csv = 'outputs/predictions.csv'
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved in {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify images with the trained CNN-Model.')
    parser.add_argument('image_path', type=str, 
                        help='Path to a single image or a directory with multiple images.')
    parser.add_argument('--show', action='store_true',
                        help='Show Images with predictions.')
    args = parser.parse_args()
    main(args)
