"""One-shot script to upload the trained waste classifier to Hugging Face Hub.
Run this after training is complete and finetuned_model.pth exists locally.
"""
from pathlib import Path
from huggingface_hub import HfApi

# Build the model path relative to this script: scripts/ sits one level below
# the package root, so parent.parent resolves to waste_recycling_chatbot_pa2/.
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "baseline" / "finetuned_model.pth"

if not MODEL_PATH.exists():
    print("Model file not found:")
    print(MODEL_PATH)
    print("Train the model first (run 02_baseline_training.ipynb), then re-run this script.")
    raise SystemExit

api = HfApi()

print("Uploading model to Hugging Face Hub...")

api.upload_file(
    path_or_fileobj=str(MODEL_PATH),
    path_in_repo="finetuned_model.pth",
    repo_id="le7lum/swiss-waste-classifier",
    repo_type="model",
    commit_message="Update classifier model with new waste category"
)

print("Done! Model uploaded to Hugging Face.")