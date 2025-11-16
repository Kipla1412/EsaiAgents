from smolagents import Tool
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import requests
import torch
from typing import Dict, Any

class MedGemmaTool(Tool):
    name = "medgemma_image_chat"
    description = "Answer questions using text or text+image (CPU/GPU-friendly)."
    inputs = {
        "text": {"type": "string", "description": "The user prompt text."},
        "image_url": {"type": "string", "description": "Optional image URL.", "nullable": True},
        "image_path": {"type": "string", "description": "Optional local image path.", "nullable": True}
    }
    output_type = "string"

    def __init__(self, model_dir: str):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_dir,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_dir)

    def forward(self, text: str, image_url: str = None, image_path: str = None) -> str:
        if not text.strip():
            return "Error: Please provide a text prompt."
        if image_url and image_path:
            return "Error: Provide only image_url OR image_path, not both."

        image = None
        if image_url:
            try:
                resp = requests.get(image_url, stream=True)
                image = Image.open(resp.raw).convert("RGB")
            except Exception as e:
                return f"Error loading image from URL: {e}"
        elif image_path:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                return f"Error loading local image: {e}"

        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        if image:
            messages[0]["content"].append({"type": "image", "image": image})

        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.device)
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                gen = self.model.generate(**inputs, max_new_tokens=150)

            return self.processor.decode(gen[0][input_len:], skip_special_tokens=True)
        except Exception as e:
            return f"Error generating response: {e}"
        
def medgemma_tool(text: str, image_url: str = None, image_path: str = None) -> str:
    """
    Mini MedGemma tool function for SmolAgents.
    """
    model_dir = r"D:\backend\txtai\src\python\txtai\model\medgemma-4b-it"  # change if your model path differs
    tool = MedGemmaTool(model_dir=model_dir)
    return tool.forward(text, image_url=image_url, image_path=image_path)
