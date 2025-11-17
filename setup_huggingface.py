"""
Setup Hugging Face authentication and use open models
"""
import os
from huggingface_hub import login, HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def setup_huggingface():
    """Setup Hugging Face authentication"""
    
    # Option 1: Use your token
    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        try:
            login(token=token)
            print("[OK] Logged in to Hugging Face")
        except:
            print("[ERROR] Invalid Hugging Face token")
    
    # Option 2: Use open models that don't require authentication
    print("\nUsing open models for training...")
    
    # Test with a smaller open model
    test_models = [
        "microsoft/phi-2",  # 2.7B parameters, no auth required
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B, very efficient
        "facebook/opt-1.3b",  # 1.3B, open model
    ]
    
    for model_name in test_models:
        try:
            print(f"Testing {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"[OK] {model_name} is accessible!")
            return model_name
        except Exception as e:
            print(f"[ERROR] {model_name} failed: {str(e)[:50]}")
    
    return None

if __name__ == "__main__":
    model = setup_huggingface()
    if model:
        print(f"\n[OK] Recommended model: {model}")
        print("Update your .env file:")
        print(f"MODEL_NAME={model}")