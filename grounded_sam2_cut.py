# 保持裁剪后的图片比例与原图一致
import os
import cv2
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "Grounded-SAM-2"))
import torch
import uuid
import torch
import requests
import numpy as np
from torchvision.ops import box_convert
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
from torchvision.ops import box_convert
# sys.path.insert(0, '/root/project/ComfyUI/custom_nodes/ComfyUI-RomanticQq/Grounded-SAM-2')
sys.path.insert(0, f'{os.path.dirname(__file__)}/Grounded-SAM-2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict

class GroundedSam2Cut:
    def __init__(self):
        self.dir_name = os.path.dirname(__file__)
        self.tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        """
        1. 配置 Hyper parameters
        """
        # 模型地址
        SAM2_CHECKPOINT = f"{self.dir_name}/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
        SAM2_MODEL_CONFIG = f"configs/sam2.1/sam2.1_hiera_l.yaml"
        GROUNDING_DINO_CONFIG = f"{self.dir_name}/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        GROUNDING_DINO_CHECKPOINT = f"{self.dir_name}/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_prompt = "subject"
        self.box_threshold = 0.35
        self.text_threshold = 0.25

        """
        2. 加载模型
        """
        # build SAM2 image predictor
        sam2_checkpoint = SAM2_CHECKPOINT
        model_cfg = SAM2_MODEL_CONFIG
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # load grounding dino model
        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=self.DEVICE
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # "image": ("IMAGE", {"default": None, "forceInput": True}),
                "text_prompt": ("STRING", {"default": "subject"}),
                "box_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "text_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
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
    DESCRIPTION = "使用grounded-sam2进行检测和分割（仅输出分割叠加图）"
    def test(self, text_prompt="subject", box_threshold=0.35, text_threshold=0.25, random_seed=0, image=None, imageUrl=None):
        np.random.seed(random_seed)
        try:
            self.text_prompt = text_prompt
            self.box_threshold = box_threshold
            self.text_threshold = text_threshold
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
            segmented_img = self.process_single_image(tmp_img_path)
            if segmented_img is not None:
                cv2.imwrite(tmp_img_path, segmented_img)
                print(f"Finished processing {tmp_img_path}")
        except Exception as e:
            print(f"Error processing image: {e}")
        img = cv2.imread(tmp_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(np.expand_dims(img, axis=0) / 255.0)
        os.remove(tmp_img_path)
        return (img,)

    def process_single_image(self, img_path):
        '''
        只进行检测和分割，返回带掩码叠加的整图（BGR）
        '''
        try:
            image_source, image = load_image(img_path)
            self.sam2_predictor.set_image(image_source)
            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=image,
                caption=self.text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            h, w, _ = image_source.shape
            if boxes is None or boxes.shape[0] == 0:
                # 没有检测到对象，返回原图
                return cv2.imread(img_path)

            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            # masks: (N, H, W) or (N,1,H,W)
            if masks is None or masks.shape[0] == 0:
                return cv2.imread(img_path)

            if masks.ndim == 4:
                masks = masks.squeeze(1)

            img = cv2.imread(img_path)  # BGR
            # 只保留分割得到的部分，其他区域用白色填充（不显示检测框）
            result = np.ones_like(img, dtype=np.uint8) * 255  # 白色背景
            num_masks = masks.shape[0]
            for i in range(num_masks):
                mask_i = masks[i]
                # Ensure mask matches image spatial shape
                if mask_i.shape != (h, w):
                    # masks 可能为 float 或 uint8，先转为 float 再 resize
                    mask_resized = cv2.resize(mask_i.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_resized = mask_i.astype(np.float32)

                # 二值化掩码（阈值 0.5）
                mask_bool = mask_resized > 0.5
                # 将原图中对应位置复制到结果图
                result[mask_bool] = img[mask_bool]

            # 使用检测得到的坐标进行裁剪，只保留检测到的区域（取所有检测框的并集）
            try:
                # input_boxes: N x 4 (xyxy)
                boxes_np = np.asarray(input_boxes, dtype=np.float32)
                if boxes_np.size == 0:
                    # 没有框则返回整图结果
                    return result

                x1s = boxes_np[:, 0]
                y1s = boxes_np[:, 1]
                x2s = boxes_np[:, 2]
                y2s = boxes_np[:, 3]

                x1 = int(max(0, np.floor(x1s.min())))
                y1 = int(max(0, np.floor(y1s.min())))
                x2 = int(min(w, np.ceil(x2s.max())))
                y2 = int(min(h, np.ceil(y2s.max())))

                # 防止无效框
                if x2 <= x1 or y2 <= y1:
                    return result

                cropped = result[y1:y2, x1:x2].copy()
                return cropped
            except Exception as e:
                print(f"Cropping error: {e}")
                return result
        except Exception as e:
            print(f"Segmentation error: {e}")
            return None
