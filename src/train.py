import numpy as np
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def train():
    """
    Trains the CNN model using the preprocessed training data with data augmentation.
    
    Saves the best model and training history to the respective directories.
    
    Returns:
        None
    """
    # Load the preprocessed data
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    y_val = np.load('data/processed/y_val.npy')
    
    # Build the model
    model = build_model()
    model.summary()
    
    # Define callbacks
    checkpoint = ModelCheckpoint('models/best_model.keras', 
                                 monitor='val_accuracy', 
                                 save_best_only=True, 
                                 mode='max',
                                 verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', 
                               patience=10, 
                               mode='max', 
                               restore_best_weights=True,
                               verbose=1)
    
    # Define data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Fit the data generator on training data
    datagen.fit(X_train)
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Train the model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=2, # Aus Testzwecken weniger
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stop]
    )
    
    # Save the training history
    np.save('outputs/history.npy', history.history)
    print("Training completed and history saved in 'outputs/history.npy'.")

if __name__ == "__main__":
    train()
