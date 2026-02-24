import os
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

def download_models():
    print("Downloading BLIP Captioning Model...")
    processor_cap = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_cap = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    os.makedirs("models/weights/blip-image-captioning-base", exist_ok=True)
    processor_cap.save_pretrained("models/weights/blip-image-captioning-base")
    model_cap.save_pretrained("models/weights/blip-image-captioning-base")
    print("Captioning Model saved locally.")

    print("Downloading BLIP VQA Model...")
    processor_vqa = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model_vqa = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    
    os.makedirs("models/weights/blip-vqa-base", exist_ok=True)
    processor_vqa.save_pretrained("models/weights/blip-vqa-base")
    model_vqa.save_pretrained("models/weights/blip-vqa-base")
    print("VQA Model saved locally.")

if __name__ == "__main__":
    download_models()
    print("All models downloaded successfully.")
