import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate():
    """
    Evaluates the trained CNN model on the validation dataset.
    
    Generates and saves classification reports, confusion matrices, and training history plots.
    
    Returns:
        None
    """
    # Load the validation data
    X_val = np.load('data/processed/X_val.npy')
    y_val = np.load('data/processed/y_val.npy')
    
    # Load the best saved model
    model = load_model('models/best_model.keras')

    # Check last layer name for explainability
    model.summary();
    
    # Evaluate the model on validation data
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f'Validation Loss: {loss:.4f}')
    print(f'Validation Accuracy: {accuracy:.4f}')
    
    # Generate predictions
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
    
    # Generate classification report
    report = classification_report(y_val, y_pred, target_names=['Negative', 'Positive'])
    print("\nClassification Report:")
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('outputs/confusion_matrix.png')
    plt.show()
    
    # Load training history
    history = np.load('outputs/history.npy', allow_pickle=True).item()
    
    # Plot Accuracy over Epochs
    plt.figure(figsize=(10,5))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('outputs/accuracy.png')
    plt.show()
    
    # Plot Loss over Epochs
    plt.figure(figsize=(10,5))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('outputs/loss.png')
    plt.show()
    
    print("Evaluation completed and plots saved in 'outputs/' directory.")

if __name__ == "__main__":
    evaluate()

