from PIL import Image
import numpy as np
import io
try:
    import clip
    import torch
    CLIP = True
except Exception:
    CLIP = False

class PerceptorV2:
    def __init__(self, emb_mod):
        self.emb = emb_mod
        self.clip_model = None
        if CLIP:
            try:
                self.clip_model, self.preproc = clip.load('ViT-B/32', device='cpu')
            except Exception:
                self.clip_model = None

    def perceive_text(self, text):
        return self.emb.encode_text(text)

    def perceive_image(self, image):
        if isinstance(image, bytes):
            import io
            image = Image.open(io.BytesIO(image)).convert('RGB')
        if self.clip_model is not None:
            img = self.preproc(image).unsqueeze(0)
            with torch.no_grad():
                emb = self.clip_model.encode_image(img).cpu().numpy().ravel().astype(np.float32)
            return emb
        return self.emb.encode_image(image)
