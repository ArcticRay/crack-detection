import os
import cv2 # For Image Processing and Computer Vision Tasks
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_dir, image_size=(128, 128)):
    """
    Loads and preprocesses images from the specified directory.

    Args:
        data_dir (str): Path to the directory containing raw image data. 
        This directory should contain subfolders “Positive” and “Negative”, which represent the respective classes.

        image_size (tuple): Desired image size for resizing (width, height).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            - images (np.ndarray): Array of preprocessed images.
            - labels (np.ndarray): Array of corresponding labels.
    """

    # Empty Lists to store images and labels
    images = []
    labels = []
    # Turn classes into numerical representation
    label_mapping = {'Positive': 1, 'Negative': 0}

    for label_name, label_value in label_mapping.items():
        label_dir = os.path.join(data_dir, label_name) # Create Folder Path for Positive and Negative Classes

        if not os.path.isdir(label_dir):
            print(f"Warning: {label_dir} does not exist. Skipping.")
            continue

        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)

            # Check for valid image extensions
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Failed to load image {img_path}. Skipping.")
                    continue

                # Resize image to desired size and add to coresponding lists
                img = cv2.resize(img, image_size)
                images.append(img)
                labels.append(label_value)

    if not images:
        raise ValueError("No images found. Please check the data directory structure.")

    # Convert lists to NumPy arrays and normalize pixel values
    images = np.array(images).reshape(-1, image_size[0], image_size[1], 1) / 255.0 # Pixel Values should be between 0 and 1
    labels = np.array(labels)
    return images, labels

def preprocess(data_dir='data/raw/', output_dir='data/processed/', test_size=0.2, random_state=42):
    """
    Preprocesses the raw image data and splits it into training and validation sets.

    Args:
        data_dir (str): Path to the directory containing raw image data.
        output_dir (str): Path to the directory where processed data will be saved.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Seed used by the random number generator.

    Returns:
        None
    """
    # Load and preprocess data
    images, labels = load_data(data_dir)
    
    # Split data into training and validation sets (with stratification (Btoh classes have 20.000 Entries))
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed data as NumPy arrays
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    
    print("Data preprocessing completed and saved to 'data/processed/'.")

if __name__ == "__main__":
    preprocess()
