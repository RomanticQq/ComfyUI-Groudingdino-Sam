import os
import cv2
import uuid
import json
import torch
import requests
import numpy as np
from torchvision.ops import box_convert
from datetime import datetime, timedelta
from groundingdino.util.inference import load_model, load_image, predict, annotate
import clip
from PIL import Image


class GroundingDinoClip2:
    def __init__(self):
        self.dir_name = os.path.dirname(__file__)
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.model = load_model(f"{self.dir_name}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", f"{self.dir_name}/GroundingDINO/weights/groundingdino_swint_ogc.pth")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_clip, self.preprocess_clip = clip.load("ViT-B/32", device=self.device)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # "image": ("IMAGE", {"default": None, "forceInput": True}),
                "text_prompt": ("STRING", {"default": "subject"}),
                "box_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "text_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "clip_threshold": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "random_seed": ("INT", {"default": 66666, "min": 0, "max": 2**32 - 1, "step": 1, "control_after_generate": True}),
            },
            "optional": {
                "image": ("IMAGE", {"default": None}),
                "imageUrl": ("STRING", {"default": None}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_TYPES_NAMES = ("image",)
    FUNCTION = "test"
    CATEGORY = "Grounding-Sam"
    DESCRIPTION = "使用groundingdino进行目标检测，并使用clip进行相似度计算"
    def test(self, text_prompt="subject", box_threshold=0.35, text_threshold=0.25, clip_threshold=0.9, random_seed=0, image=None, imageUrl=None):
        text_prompt = text_prompt.lower()
        np.random.seed(random_seed)
        try:
            tmp_img_name = str(uuid.uuid4()) + ".jpg"
            tmp_img_path = os.path.join(self.tmp_dir, tmp_img_name)
            if image is not None:
                img = image.numpy()[0]
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(tmp_img_path, img)
            elif imageUrl is not None:
                response = requests.get(imageUrl)
                if response.status_code == 200:
                    with open(tmp_img_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Image downloaded from {imageUrl} and saved to {tmp_img_path}")
                else:
                    raise ValueError(f"Failed to download image from {imageUrl}, status code: {response.status_code}")
            else:
                raise ValueError("Either 'image' or 'imageUrl' must be provided.")
            print(f"Processing image {tmp_img_path} with prompt '{text_prompt}'")

            boxes, logits, phrases, image_source = self.step1_jiance(tmp_img_path, text_prompt, box_threshold, text_threshold)
            # if boxes is None or boxes.shape[0] == 0:
            #     raise ValueError("No objects found in image")
            new_boxes = []
            new_logits = []
            new_phrases = []
            h, w, _ = image_source.shape
            wh_boxes = boxes * torch.Tensor([w, h, w, h])
            cut_img_paths = []
            input_boxes = box_convert(boxes=wh_boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            for i, box in enumerate(input_boxes):
                cut_img_name = str(uuid.uuid4()) + ".jpg"
                cut_img_path = os.path.join(self.tmp_dir, cut_img_name)
                x1, y1, x2, y2 = [int(num) for num in box]
                origin_image = cv2.imread(tmp_img_path)
                cut_image = origin_image[y1:y2, x1:x2]
                cv2.imwrite(cut_img_path, cut_image)
                cut_img_paths.append(cut_img_path)

            clip_prompts = [t.strip() for t in text_prompt.split(".")]
            scores = self.step2_clip(cut_img_paths, clip_prompts)
            print("clip的结果",scores)

            for i, score in enumerate(scores):
                if score > clip_threshold:
                    new_boxes.append(boxes[i])
                    new_logits.append(logits[i])
                    new_phrases.append(phrases[i])
            if len(new_boxes) > 0:
                new_boxes = torch.stack(new_boxes)
            if len(new_logits) > 0:
                new_logits = torch.stack(new_logits)
            if len(new_boxes) > 0:
                annotated_frame = annotate(image_source=image_source, boxes=new_boxes, logits=new_logits, phrases=new_phrases)
                cv2.imwrite(tmp_img_path, annotated_frame)
        except Exception as e:
            print(f"Error processing image: {e}")
        img = cv2.imread(tmp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        os.remove(tmp_img_path)
        # return (image,)
        return (img,)
    def step1_jiance(self, IMAGE_PATH, TEXT_PROMPT,BOX_TRESHOLD,TEXT_TRESHOLD):
        image_source, image = load_image(IMAGE_PATH)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        print(phrases)
        print(logits)
        print(boxes)
        return boxes, logits, phrases, image_source
    def step2_clip(self, img_paths, text_prompt):
        image = [self.preprocess_clip(Image.open(img_path)).unsqueeze(0).to(self.device) for img_path in img_paths]
        image = torch.cat(image, dim=0)
        text = clip.tokenize(text_prompt).to(self.device)
        with torch.no_grad():
            image_features = self.model_clip.encode_image(image)
            text_features = self.model_clip.encode_text(text)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            scores = image_features @ text_features.t()
            scores = torch.diag(scores)
            scores = scores.cpu().numpy()
        return scores
