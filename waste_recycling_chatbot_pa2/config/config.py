"""
Central configuration for all project paths
"""
from pathlib import Path

# ============================================================================
# PROJECT ROOT
# ============================================================================
# Old Project Root
#PROJECT_ROOT = Path(r"C:\ZHAW\HS25\PA2\waste_recycling_chatbot_pa2")

# New Project Root
PROJECT_ROOT = Path(r"C:\Users\Lejlum\Documents\PA2_Recycling_Chatbot\waste_recycling_chatbot_pa2")

# ============================================================================
# ALL OTHER PATHS
# ============================================================================

# Archive (Backup of original files)
ARCHIVE_ROOT = PROJECT_ROOT.parent / "_archive_original"  # Eine Ebene höher
REALWASTE_ORIGINAL = ARCHIVE_ROOT / "realwaste-main" / "RealWaste"
ECOVISION_ORIGINAL = ARCHIVE_ROOT / "ecovision_mobilenetv3"

# Data
DATA_ROOT = PROJECT_ROOT.parent / "data"
RAW_DATA = DATA_ROOT / "raw"
PROCESSED_DATA = DATA_ROOT / "processed" / "organized_dataset"

# Waste Chatbot Dataset (raw + labels)
WASTE_CHATBOT_DATASET = DATA_ROOT / "waste_chatbot_dataset"
WASTE_CHATBOT_EXCEL = DATA_ROOT / "waste_chatbot_excel.xlsx"

# Output für neu organisiertes Dataset
PROCESSED_NEW = DATA_ROOT / "processed_new" / "organized_dataset"

# Models
MODEL_ROOT = PROJECT_ROOT / "models"
BASELINE_MODEL = PROJECT_ROOT / "_archive_original" / "ecovision_mobilenetv3" / "pytorch_model.bin"  # Original (READ-ONLY)
FINETUNED_MODEL = MODEL_ROOT / "baseline" / "finetuned_model.pth"  # Ihr trainiertes Model
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
FINAL_MODEL_DIR = MODEL_ROOT / "baseline"

# Outputs
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = OUTPUT_ROOT / "logs"
REPORTS_DIR = OUTPUT_ROOT / "reports"

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ============================================================================
# CONSTANTS
# ============================================================================

ECOVISION_CLASSES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

# Training hyperparameters (optional, can be used later)
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 20,
    'early_stopping_patience': 5,
    'train_ratio': 0.70,
    'val_ratio': 0.15
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_paths():
    """Check if all important paths exist"""
    print("Checking paths...")
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Exists: {'YES' if PROJECT_ROOT.exists() else 'NO'}")
    
    print(f"\nRealWaste Original: {REALWASTE_ORIGINAL}")
    print(f"Exists: {'YES' if REALWASTE_ORIGINAL.exists() else 'NO'}")
    
    print(f"\nProcessed Data: {PROCESSED_DATA}")
    print(f"Exists: {'YES' if PROCESSED_DATA.exists() else 'NO'}")
    
    print(f"\nModels: {MODEL_ROOT}")
    print(f"Exists: {'YES' if MODEL_ROOT.exists() else 'NO'}")
    
    print(f"\nOutputs: {OUTPUT_ROOT}")
    print(f"Exists: {'YES' if OUTPUT_ROOT.exists() else 'NO'}")


def create_directories():
    """Create all necessary directories"""
    dirs_to_create = [
        DATA_ROOT, RAW_DATA, PROCESSED_DATA,
        MODEL_ROOT, CHECKPOINT_DIR, FINAL_MODEL_DIR,
        OUTPUT_ROOT, FIGURES_DIR, LOGS_DIR, REPORTS_DIR,
        NOTEBOOKS_DIR
    ]
    
    print("Creating directories...")
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path.relative_to(PROJECT_ROOT)}")
    
    print("\nAll directories created successfully!")


# Auto-check on import (optional)
if __name__ == "__main__":
    check_paths()

