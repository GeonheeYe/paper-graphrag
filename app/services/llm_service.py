from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_name):
    # 모델 로드    
    print("Loading Vision-language model...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cuda:0",
    ).to("cuda")

    print("Vision-language model loaded successfully!")
    return processor, model

def load_hf_model(model_path: str, device: str = "cuda:0"):
    """HuggingFace 모델 로드"""
    print(f"Loading HuggingFace model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
    ).to("cuda")
    model.eval()
    return model, tokenizer