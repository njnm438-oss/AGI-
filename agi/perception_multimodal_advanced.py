from typing import Any, Dict
import numpy as np
from PIL import Image
import io
import logging
import time

logger = logging.getLogger("agi.perception")

class AdvancedMultimodalPerception:
    def __init__(self, embedding_module=None):
        self.embedding = embedding_module
        self.has_clip = False
        try:
            import clip  # type: ignore
            import torch
            self.clip = clip
            self.torch = torch
            self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device="cpu")
            self.has_clip = True
            logger.info("CLIP loaded for perception")
        except Exception as e:
            logger.info(f"CLIP not available, fallback enabled: {e}")

    def _load_image(self, image: Any):
        if image is None:
            raise ValueError("Image is None")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image.astype("uint8")).convert("RGB")
        raise TypeError("Unsupported image type")

    def perceive_scene(self, image: Any, context: str = "") -> Dict[str, Any]:
        try:
            image = self._load_image(image)
        except Exception as e:
            logger.error(f"Invalid image: {e}")
            return {"image_embedding": np.zeros(32, dtype=np.float32), "description": "invalid image", "raw_image": None}
        if self.has_clip:
            try:
                img = self._clip_preprocess(image).unsqueeze(0)
                with self.torch.no_grad():
                    emb = self._clip_model.encode_image(img).cpu().numpy().squeeze()
                emb = emb.astype(np.float32)
                desc = f"Scene ({context}): CLIP embedding"
                return {"image_embedding": emb, "description": desc, "raw_image": image}
            except Exception as e:
                logger.warning(f"CLIP failed: {e}")
        arr = np.array(image).astype(np.float32)/255.0
        mean_rgb = arr.mean(axis=(0,1))
        hist = np.histogram(arr.flatten(), bins=16, range=(0,1))[0].astype(np.float32)
        emb = np.concatenate([mean_rgb, hist])
        emb /= (np.linalg.norm(emb) + 1e-9)
        desc = f"Scene ({context}): fallback image features"
        return {"image_embedding": emb.astype(np.float32), "description": desc, "raw_image": image}

    def vision_to_knowledge_graph(self, scene_info: Dict[str, Any], kg):
        desc = scene_info.get("description", "")
        node_id = f"vision_{abs(hash(desc)) % 10_000_000}"
        data = {"description": desc, "timestamp": time.time(), "source": "vision"}
        try:
            if hasattr(kg, "add_node"):
                kg.add_node(node_id, data)
            elif hasattr(kg, "add"):
                kg.add(node_id, data)
            else:
                logger.error("KG missing add_node/add")
        except Exception as e:
            logger.error(f"Failed update KG: {e}")
