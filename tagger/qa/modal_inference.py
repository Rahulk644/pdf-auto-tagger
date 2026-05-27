"""
Modal inference app for QA auditing using Gemma 4 27B IT (multimodal, fp16).

The public HuggingFace weights are google/gemma-4-27b-it — the open-weight
equivalent of the gemma-4-31b-it model served via Google's Gemini API.
If Google releases 31B weights separately, update MODEL_ID and MODEL_DIR.

Deploy:
    modal deploy tagger/qa/modal_inference.py

After deployment, app_auditor_modal.py references it by name:
    modal.Cls.from_name("qa-gemma4-inference", "GemmaInference")
"""
import modal

VOLUME_NAME = "gemma4-weights"
MODEL_ID = "google/gemma-4-31B-it"
MODEL_DIR = "/weights/gemma-4-31B-it"

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("packaging")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers>=4.51.0",
        "accelerate>=0.30.0",
        "Pillow",
        "huggingface_hub",
        "sentencepiece",
    )
)

app = modal.App("qa-gemma4-inference")


@app.cls(
    image=image,
    gpu="H100",
    volumes={"/weights": vol},
    timeout=600,
    scaledown_window=300,
    max_containers=9,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class GemmaInference:
    @modal.enter()
    def load_model(self):
        import os
        import torch
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration
        from huggingface_hub import snapshot_download

        if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
            print(f"Downloading {MODEL_ID} to {MODEL_DIR}...")
            snapshot_download(
                MODEL_ID,
                local_dir=MODEL_DIR,
                token=os.environ.get("HF_TOKEN"),
            )
            vol.commit()
            print("Download complete. Weights cached in volume.")

        print(f"Loading model from {MODEL_DIR}...")
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR)
        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()
        print("Model ready.")

    @modal.method()
    def generate(self, image_b64: str, prompt: str) -> str:
        import base64
        import io
        import torch
        from PIL import Image

        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=16500,
                temperature=0.1,
                do_sample=True,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated_ids, skip_special_tokens=True)
