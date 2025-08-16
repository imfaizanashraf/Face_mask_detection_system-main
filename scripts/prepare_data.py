import os
import glob
import random
import pandas as pd
import shutil
from PIL import Image
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(42)

# Updated Paths - using relative paths from the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

IMAGE_ROOT = os.path.join(PROJECT_ROOT, 'data', 'images', 'Images')
LABEL_ROOT = os.path.join(PROJECT_ROOT, 'data', 'labels', 'PSPI')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data')
BACKUP_DIR = os.path.join(PROJECT_ROOT, 'data', 'backup')

def check_directories():
    """Check if required directories exist and create them if needed."""
    directories = {
        'Image Root': IMAGE_ROOT,
        'Label Root': LABEL_ROOT,
        'Output Directory': OUTPUT_DIR,
        'Backup Directory': BACKUP_DIR
    }
    
    for name, path in directories.items():
        if not os.path.exists(path):
            logger.warning(f"{name} does not exist at: {path}")
            if name in ['Output Directory', 'Backup Directory']:
                os.makedirs(path, exist_ok=True)
                logger.info(f"Created {name} at: {path}")
            else:
                raise FileNotFoundError(f"{name} not found at: {path}")

def validate_image(image_path):
    """Validate if image can be opened and is valid."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image {image_path}: {str(e)}")
        return False

def validate_label(label_path):
    """Validate if label file exists and contains valid data."""
    try:
        if not os.path.exists(label_path):
            return False
        with open(label_path, 'r') as f:
            value = float(f.readline().strip().split()[0])
            return 0 <= value <= 16  # PSPI range is 0-16
    except Exception as e:
        logger.error(f"Invalid label {label_path}: {str(e)}")
        return False

def create_backup():
    """Create backup of existing data files."""
    if os.path.exists(OUTPUT_DIR):
        backup_path = os.path.join(BACKUP_DIR, f"backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(backup_path, exist_ok=True)
        for file in ['train.csv', 'val.csv', 'test.csv']:
            if os.path.exists(os.path.join(OUTPUT_DIR, file)):
                shutil.copy2(
                    os.path.join(OUTPUT_DIR, file),
                    os.path.join(backup_path, file)
                )
        logger.info(f"Created backup at {backup_path}")

def collect_pairs(image_root, label_root):
    """Collect valid image-label pairs with validation."""
    image_paths = glob.glob(os.path.join(image_root, '**', '*.png'), recursive=True)
    pairs = []
    invalid_pairs = []
    
    logger.info(f"Found {len(image_paths)} total images")
    
    if len(image_paths) == 0:
        logger.error(f"No images found in {image_root}")
        logger.info("Please ensure your images are in the correct directory structure:")
        logger.info(f"Expected path: {image_root}")
        return []
    
    for img_path in image_paths:
        rel_path = os.path.relpath(img_path, image_root)
        label_path = os.path.join(label_root, rel_path).replace('.png', '_facs.txt')
        
        if validate_image(img_path) and validate_label(label_path):
            pairs.append((img_path, label_path))
        else:
            invalid_pairs.append((img_path, label_path))
    
    logger.info(f"Valid pairs: {len(pairs)}")
    logger.info(f"Invalid pairs: {len(invalid_pairs)}")
    
    if invalid_pairs:
        logger.warning("Some image-label pairs were invalid. Check the logs for details.")
    
    return pairs

def analyze_dataset(pairs):
    """Analyze the dataset distribution."""
    if not pairs:
        logger.warning("No valid pairs to analyze")
        return
        
    pain_values = []
    for _, label_path in pairs:
        with open(label_path, 'r') as f:
            value = float(f.readline().strip().split()[0])
            pain_values.append(value)
    
    df = pd.DataFrame({'pain_value': pain_values})
    stats = df['pain_value'].describe()
    
    logger.info("\nDataset Statistics:")
    logger.info(f"Mean pain value: {stats['mean']:.2f}")
    logger.info(f"Std pain value: {stats['std']:.2f}")
    logger.info(f"Min pain value: {stats['min']:.2f}")
    logger.info(f"Max pain value: {stats['max']:.2f}")
    
    # Create histogram of pain values
    plt.figure(figsize=(10, 6))
    plt.hist(pain_values, bins=20, edgecolor='black')
    plt.title('Distribution of Pain Values')
    plt.xlabel('Pain Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(OUTPUT_DIR, 'pain_distribution.png'))
    plt.close()

def main():
    """Main function to prepare the dataset."""
    try:
        # Check directories
        check_directories()
        
        # Create backup of existing data
        create_backup()
        
        # Collect and validate pairs
        pairs = collect_pairs(IMAGE_ROOT, LABEL_ROOT)
        
        if not pairs:
            logger.error("No valid image-label pairs found. Please check your data directories.")
            return
            
        random.shuffle(pairs)
        
        # Split dataset
        n_total = len(pairs)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        n_test = n_total - n_train - n_val
        
        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:n_train+n_val]
        test_pairs = pairs[n_train+n_val:]
        
        # Save splits
        def save_csv(pairs, filename):
            df = pd.DataFrame(pairs, columns=['image_path', 'label_path'])
            df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
            logger.info(f"Saved {filename} with {len(pairs)} pairs")
        
        save_csv(train_pairs, 'train.csv')
        save_csv(val_pairs, 'val.csv')
        save_csv(test_pairs, 'test.csv')
        
        # Analyze dataset
        analyze_dataset(pairs)
        
        logger.info("\nDataset Split Summary:")
        logger.info(f"Total pairs: {n_total}")
        logger.info(f"Train: {len(train_pairs)} ({len(train_pairs)/n_total:.1%})")
        logger.info(f"Validation: {len(val_pairs)} ({len(val_pairs)/n_total:.1%})")
        logger.info(f"Test: {len(test_pairs)} ({len(test_pairs)/n_total:.1%})")
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
