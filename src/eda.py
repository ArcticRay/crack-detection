import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_sample_images(data_dir, label_name, num_samples=5):
    """
    Loads a specified number of sample images (5) from a given label directory.

    Args:
        data_dir (str): Path to the directory containing raw image data.
        label_name (str): Label name ('Positive' or 'Negative') indicating the subdirectory.
        num_samples (int): Number of sample images to load.

    Returns:
        List[np.ndarray]: List of loaded images.
    """
    label_dir = os.path.join(data_dir, label_name)
    images = []
    for img_file in os.listdir(label_dir)[:num_samples]:
        img_path = os.path.join(label_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def plot_sample_images():
    """
    Plots sample images from both 'Positive' and 'Negative' classes and saves the plot.
    
    Returns:
        None
    """
    data_dir = 'data/raw/'
    positive_images = load_sample_images(data_dir, 'Positive')
    negative_images = load_sample_images(data_dir, 'Negative')

    # Initialize the plot
    plt.figure(figsize=(10, 5))
    
    # Plot Positive Images
    for i, img in enumerate(positive_images):
        plt.subplot(2, 5, i+1)
        plt.imshow(img, cmap='gray')
        plt.title('Positive')
        plt.axis('off')

    # Plot Negative Images
    for i, img in enumerate(negative_images):
        plt.subplot(2, 5, i+6)
        plt.imshow(img, cmap='gray')
        plt.title('Negative')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('outputs/sample_images.png')
    plt.show()

def plot_label_distribution():
    """
    Plots the distribution of labels in both training and validation datasets and saves the plots.
    
    Returns:
        None
    """
    y_train = np.load('data/processed/y_train.npy')
    y_val = np.load('data/processed/y_val.npy')

    # Plot distribution for Training Labels
    plt.figure(figsize=(6,4))
    sns.countplot(x=y_train)
    plt.title('Distribution of Training Labels')
    plt.xlabel('Label (0=Negative, 1=Positive)')
    plt.ylabel('Count')
    plt.savefig('outputs/training_label_distribution.png')
    plt.show()

    # Plot distribution for Validation Labels
    plt.figure(figsize=(6,4))
    sns.countplot(x=y_val)
    plt.title('Distribution of Validation Labels')
    plt.xlabel('Label (0=Negative, 1=Positive)')
    plt.ylabel('Count')
    plt.savefig('outputs/validation_label_distribution.png')
    plt.show()



def main():
    """
    Main function to execute EDA tasks: plotting sample images and label distributions.
    
    Returns:
        None
    """
    plot_sample_images()
    plot_label_distribution()
    print("EDA completed and plots saved in 'outputs/' directory.")

if __name__ == "__main__":
    main()
