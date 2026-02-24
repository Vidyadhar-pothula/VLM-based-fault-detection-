import torch
import os
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from PIL import Image

class BLIPLoader:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Loading BLIP models on {self.device}...")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        caption_path = os.path.join(base_dir, "models/weights/blip-image-captioning-base")
        vqa_path = os.path.join(base_dir, "models/weights/blip-vqa-base")
        
        if not os.path.exists(caption_path) or not os.path.exists(vqa_path):
             print("Warning: Local weights not found. Falling back to HuggingFace hub (this may take a while).")
             print("Run 'python download_models.py' to cache them locally for faster load times.")
             caption_path = "Salesforce/blip-image-captioning-base"
             vqa_path = "Salesforce/blip-vqa-base"

        # We can use the same processor for both
        self.processor = BlipProcessor.from_pretrained(caption_path)
        
        # Load captioning model
        self.caption_model = BlipForConditionalGeneration.from_pretrained(caption_path).to(self.device)
        self.caption_model.eval()
        
        # Load VQA model
        self.vqa_processor = BlipProcessor.from_pretrained(vqa_path)
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(vqa_path).to(self.device)
        self.vqa_model.eval()
        print("Models loaded successfully.")

    def get_image_embeddings(self, image: Image.Image):
        """
        Extracts patch embeddings from the ViT backbone of the BLIP model.
        Returns:
            patch_embeddings (torch.Tensor): Shape (1, num_patches, hidden_size)
            global_embedding (torch.Tensor): Shape (1, hidden_size)
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # The vision model within BLIP returns a BaseModelOutputWithPooling
            vision_outputs = self.caption_model.vision_model(**inputs)
            
            # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
            # The sequence length is 1 (CLS token) + num_patches
            last_hidden_state = vision_outputs.last_hidden_state
            
            # The cls_token is usually the first token
            global_embedding = last_hidden_state[:, 0, :]
            
            # The patch embeddings are the remaining tokens
            patch_embeddings = last_hidden_state[:, 1:, :]
            
        return patch_embeddings, global_embedding

    def generate_caption(self, image: Image.Image, text_prompt: str = ""):
        """
        Generates a caption for the image. Assumes prompt could guide captioning.
        """
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.caption_model.generate(**inputs, max_new_tokens=50)
        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def answer_question(self, image: Image.Image, question: str):
        """
        Answers a question about the image using the VQA model.
        """
        inputs = self.vqa_processor(images=image, text=question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vqa_model.generate(**inputs, max_new_tokens=50)
        return self.vqa_processor.decode(outputs[0], skip_special_tokens=True)
