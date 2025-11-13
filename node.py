from .groundingdino_cut import GroundingDino
from .grounded_sam2_cut_gaussian import GroundedSam2CutGaussian
from .grounded_sam2_cut import GroundedSam2Cut
from .groundingdino_draw_bbox import GroundingDinoDrawBbox


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "groundingdino": GroundingDino,
    "grounded_sam2_cut_gaussian": GroundedSam2CutGaussian,
    "grounded_sam2_cut": GroundedSam2Cut,
    "groundingdino_draw_bbox": GroundingDinoDrawBbox,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "groundingdino": "GroundingDino",
    "grounded_sam2_cut_gaussian": "GroundedSam2CutGaussian",
    "grounded_sam2_cut": "GroundedSam2Cut",
    "groundingdino_draw_bbox": "GroundingDinoDrawBbox",
}