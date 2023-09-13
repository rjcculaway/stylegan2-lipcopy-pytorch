import sys
import os
import glob

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "landmark_detection"))
sys.path.insert(
    1,
    os.path.join(
        os.path.dirname(__file__),
        "landmark_detection/hrnet_lms/experiments/300w/face_alignment_300w_hrnet_w18.yaml",
    ),
)
from landmark_detection import HRNet_lms
from PIL import Image


class LipDetector:
    LEFT_LIP_CORNER: int = 48
    RIGHT_LIP_CORNER: int = 64

    def __init__(self) -> None:
        self.model = HRNet_lms()
        return

    def load_image(self, image_path) -> np.ndarray:
        img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
        img = img / 255.0 * 2 - 1  # Map values to -1.0 to 1.0

        print(f"[{img.min(), img.max()}]")
        return img

    def preprocess_image(self, img) -> torch.Tensor:
        preprocessed_img = (
            torch.from_numpy(img).to("cuda:0").permute(2, 0, 1)
        )  # RGB -> BRG
        preprocessed_img = preprocessed_img.unsqueeze(0)  # Add an additional dimension
        pool = nn.AvgPool2d(4, 4)
        preprocessed_img = pool(
            preprocessed_img
        )  # Downsample to 25% by average pooling

        return preprocessed_img

    def preprocess_image_from_tensor(self, img) -> torch.Tensor:
        pool = nn.AvgPool2d(4, 4)
        preprocessed_img = pool(img)  # Downsample to 25% by average pooling

        return preprocessed_img

    def process_image(self, path):
        return self.preprocess_image(self.load_image(path))

    def detect_lips(self, preprocessed_img):
        heatmaps = self.model(preprocessed_img)
        # marks, heatmap_grid = self.model.parse_heatmaps(heatmaps[0], (256, 256))

        return heatmaps[
            :, [self.LEFT_LIP_CORNER, 51, 57, 62, 66, self.RIGHT_LIP_CORNER]
        ]

    def save_image_with_marks(self, img, mark_group, heatmap_grid, name_index=""):
        np_img = (img + 1) / 2 * 255  # -> Map image back to 0-255
        np_img = cv2.resize(np_img, (256, 256), interpolation=cv2.INTER_AREA)

        self.model.draw_marks(np_img, mark_group)
        cv2.imwrite("sample/results" + name_index + ".png", np_img[:, :, ::-1])
        cv2.imwrite("sample/heatmap" + name_index + ".png", heatmap_grid * 255)


if __name__ == "__main__":
    lip_detector = LipDetector()
    img_paths = glob.glob("sample/*.png")

    for i, path in enumerate(img_paths):
        img = lip_detector.process_image(path)
        lip_coordinate = lip_detector.detect_lips(img)
        print(lip_coordinate[0])
