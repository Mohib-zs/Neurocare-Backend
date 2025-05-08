import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

def download_mistral():
    # Load environment variables
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not token:
        print("Error: HUGGINGFACE_TOKEN not found in .env file")
        return
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Download Mistral model
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    local_dir = os.path.join(models_dir, "mistral-7b-instruct-v0.2")
    
    print(f"Downloading Mistral model to {local_dir}...")
    
    # Download model files
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token
    )
    
    print("Model downloaded successfully!")
    
    # Test loading the model
    print("Testing model loading...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_dir, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            torch_dtype="auto",
            device_map="auto",
            token=token
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    download_mistral() 