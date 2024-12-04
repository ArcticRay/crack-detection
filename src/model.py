import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def build_model(input_shape=(128, 128, 1)):
    """
    Builds and compiles the CNN model for surface crack detection.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        1 Channel for Grayscale Images

    Returns:
        tensorflow.keras.models.Sequential: Compiled CNN model.
    """
    # Sequential Model: a linear stack model in which each layer has exactly one input and one output
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'), # 64 Filters for more complex "FEatures"
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'), # Further increase in  number of filters for deeper feature extraction
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        Flatten(), # Converts the multidimensional feature maps into a one dimensional vector that serves as input for the fully connected layers
        Dense(128, activation='relu'),
        Dropout(0.5), # To prevent overfitting
        # Binary Classification
        # Crack or no Crack
        Dense(1, activation='sigmoid')  # Sigmoid for probabilty 
    ])
    
    # Compile the model with optimizer, loss function, and metrics
    model.compile(optimizer='adam', # Adaptive Moment Estimation for faster convergence
                  loss='binary_crossentropy', # Loss function for binary classification problems
                  metrics=['accuracy']) # Accuracy for evaluation is fine because of 50 / 50 data split (Positive / Negative)
    
    return model
