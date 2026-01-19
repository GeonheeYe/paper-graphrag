from transformers import AutoProcessor, AutoModelForImageTextToText
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
