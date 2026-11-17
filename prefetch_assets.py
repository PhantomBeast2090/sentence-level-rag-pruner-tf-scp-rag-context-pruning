"""Pre-fetch all HuggingFace assets so the experiment can run fully offline."""
import sys
from datasets import load_dataset
from huggingface_hub import snapshot_download

reader = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
load_dataset("rajpurkar/squad_v2", split="validation")
snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
snapshot_download(reader, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "merges.txt"])
print("prefetch complete")
