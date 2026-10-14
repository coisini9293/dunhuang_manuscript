import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.rejoin.load import build_rejoin_models, load_rejoin_checkpoint
from utils.device import pick_device
from utils.rejoin_data import to_tensor


class RejoinPipeline:
    """缀合推理流水线。"""

    def __init__(self, model_path: str | Path | None = None, device: str = "auto"):
        self.device = pick_device(device)
        ckpt_path = Path(model_path or "/path/to/dunhuang-autohdr/MODELS/weights/rejoin.pth")

        if ckpt_path.exists():
            (self.backbone, self.classifier, self.pose_regressor), self.model_config = (
                load_rejoin_checkpoint(ckpt_path, self.device)
            )
        else:
            self.backbone, self.classifier, self.pose_regressor, self.model_config = (
                build_rejoin_models(pretrained=True, device=self.device)
            )

        self.backbone.eval()
        self.classifier.eval()
        self.pose_regressor.eval()

    @torch.no_grad()
    def _tensor_from_bgr(self, image_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return to_tensor(rgb).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_pair(self, img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> tuple[float, np.ndarray]:
        feat1 = self.backbone(self._tensor_from_bgr(img1_bgr))
        feat2 = self.backbone(self._tensor_from_bgr(img2_bgr))
        confidence = self.classifier(feat1, feat2).item()
        pose = self.pose_regressor(feat1, feat2).squeeze().cpu().numpy()
        return confidence, pose
