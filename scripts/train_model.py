import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'app', 'model', 'model.pt')

# Create model directory if it doesn't exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
IMG_SIZE = 224
PATIENCE = 5

# Dataset
class PainDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        self.data = pd.read_csv(csv_path)
        if len(self.data) == 0:
            raise ValueError(f"Empty dataset file: {csv_path}")
        self.transform = transform
        logger.info(f"Loaded dataset from {csv_path} with {len(self.data)} samples")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        label_path = self.data.iloc[idx]['label_path']
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label not found: {label_path}")
            
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            label = float(lines[0].strip().split()[0])
            
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# DataLoaders
def get_loader(split):
    csv_path = os.path.join(DATA_DIR, f'{split}.csv')
    return DataLoader(
        PainDataset(csv_path, transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=(split=='train'),
        num_workers=2
    )

def train_model():
    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Load data
        logger.info("Loading datasets...")
        train_loader = get_loader('train')
        val_loader = get_loader('val')
        test_loader = get_loader('test')

        # Model
        logger.info("Initializing model...")
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model = model.to(device)

        criterion = nn.L1Loss()  # MAE
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        def calculate_metrics(outputs, labels, threshold=0.5):
            predictions = (outputs > threshold).float()
            accuracy = accuracy_score(labels.cpu(), predictions.cpu())
            return accuracy

        # Training loop with early stopping
        logger.info("Starting training...")
        best_val_mae = float('inf')
        patience_counter = 0

        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_losses = []
            train_accuracies = []
            
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                train_acc = calculate_metrics(outputs, labels)
                train_accuracies.append(train_acc)
            
            avg_train_loss = np.mean(train_losses)
            avg_train_acc = np.mean(train_accuracies)
            
            # Validation
            model.eval()
            val_losses = []
            val_accuracies = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.float().unsqueeze(1).to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_losses.append(loss.item())
                    val_acc = calculate_metrics(outputs, labels)
                    val_accuracies.append(val_acc)
            
            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accuracies)
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(avg_train_acc)
            history['val_acc'].append(avg_val_acc)
            
            logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}")
            logger.info(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_mae:
                best_val_mae = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), MODEL_PATH)
                logger.info("Model saved.")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logger.info("Early stopping triggered.")
                    break

        # Final evaluation on test set
        logger.info("\nEvaluating on test set...")
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                test_preds.extend((outputs > 0.5).float().cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # Calculate final metrics
        test_preds = np.array(test_preds).flatten()
        test_labels = np.array(test_labels).flatten()

        logger.info("\nFinal Test Results:")
        logger.info(f"Accuracy: {accuracy_score(test_labels, test_preds):.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(test_labels, test_preds)
        logger.info("\nConfusion Matrix:")
        logger.info(cm)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(DATA_DIR, 'confusion_matrix.png'))
        plt.close()

        # Classification Report
        logger.info("\nClassification Report:")
        logger.info(classification_report(test_labels, test_preds))

        # Plot training history
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, 'training_history.png'))
        plt.close()

        logger.info("\nTraining complete!")
        logger.info(f"Best validation MAE: {best_val_mae:.4f}")
        logger.info(f"Training history plot saved as 'training_history.png'")
        logger.info(f"Confusion matrix plot saved as 'confusion_matrix.png'")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
