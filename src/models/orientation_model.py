import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

class OrientationModel:
    def __init__(self, input_dim: int = 12, num_classes: int = 8):
        self.model = self._build_model(input_dim, num_classes)
        self.history = None
        
    def _build_model(self, input_dim: int, num_classes: int) -> Sequential:
        """Build the DNN model architecture."""
        model = Sequential([
            Dense(13, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              batch_size: int = 16, epochs: int = 200) -> Dict[str, Any]:
        """Train the model."""
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        return self.model.predict(X)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return np.argmax(self.predict(X), axis=1)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                label_mapping: Dict[int, str]) -> Dict[str, Any]:
        """Evaluate the model and return metrics."""
        y_pred = self.predict_classes(X_test)
        
        # Calculate metrics
        metrics = {
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'label_mapping': label_mapping
        }
        
        return metrics
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            raise ValueError("Model has not been trained yet")
            
        # Plot accuracy
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, metrics: Dict[str, Any]):
        """Plot confusion matrix."""
        plt.figure(figsize=(7, 7))
        plot_confusion_matrix(
            conf_mat=metrics['confusion_matrix'],
            class_names=list(metrics['label_mapping'].values()),
            show_normed=True
        )
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the model to disk."""
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'OrientationModel':
        """Load a saved model from disk."""
        model = cls()
        model.model = tf.keras.models.load_model(filepath)
        return model 