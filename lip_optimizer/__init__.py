from numpy import ndarray
import numpy as np
import torch
from torch import optim, nn, Tensor
from torchvision import utils
from tqdm import tqdm

from model import Generator
from detect_lips import LipDetector


# LipOptimizer uses landmark loss to guide optimization
class LipOptimizer:
    def __init__(self, generator: Generator, lip_detector: LipDetector, generator_args):
        self.DEVICE = "cuda:0"

        self.generator = generator
        self.lip_detector = lip_detector

        # Editing direction for the mouth
        self.mouth_editing_direction = torch.load("editing_directions/smile.pt").to(
            self.DEVICE
        )

        self.generator_args = generator_args

        self.num_iterations = 300

        self.mse_loss = nn.MSELoss(reduction="mean")
        self.l1_loss = nn.L1Loss()

        return

    def generate_from_sample_z(self, sample_z):
        """Given a latent code in Z space, generates an image."""
        self.generator.eval()

        sample, _ = self.generator(
            [sample_z.to("cuda")],
            truncation=self.generator_args.truncation,
            truncation_latent=None,
        )
        return sample

    def generate_from_sample_w(self, sample_w):
        """Given a latent code in W space, generates an image."""
        self.generator.eval()

        sample, _ = self.generator(
            [sample_w.to("cuda")],
            truncation=self.generator_args.truncation,
            truncation_latent=None,
            input_is_latent=True,
        )
        return sample

    def optimize_smile(self, source_latent: Tensor):
        """Given an input tensor in Z, changes the face such that is smiles while retaining lip expression dynamics."""
        # Convert source latent in Z to W
        source_w = self.generator.style(source_latent).detach()

        delta_latent = torch.zeros_like(source_w).to(self.DEVICE)
        delta_latent.requires_grad = True

        # Instantiate optimizer (ADAM) as well as loss functions, delta_latent and direction_magnitude
        optimizer = optim.Adam([delta_latent])
        mse_loss = nn.MSELoss(reduction="sum")

        # Generate tensor of unoptimized original image
        original_image = self.generate_from_sample_w(source_w).detach()
        original_heatmap = self.lip_detector.detect_lips(
            self.lip_detector.preprocess_image_from_tensor(original_image)
        ).detach()

        utils.save_image(
            original_image,
            f"sample/original_image.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

        # Generate tensor for unoptimized smiling image
        utils.save_image(
            self.generate_from_sample_w(
                source_w + self.mouth_editing_direction
            ).detach(),
            f"sample/smiling_image.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

        # Create dictionary for lambdas
        loss_parameters = {}
        loss_parameters["lambda_landmark_loss"] = 4000.0
        loss_parameters["lambda_smile_loss"] = 500.0

        for i in tqdm(range(self.num_iterations)):
            optimizer.zero_grad()

            current_image = self.generate_from_sample_w(
                source_w + delta_latent + self.mouth_editing_direction
            )
            current_heatmap = self.lip_detector.detect_lips(
                self.lip_detector.preprocess_image_from_tensor(current_image)
            )

            landmark_loss = mse_loss(original_heatmap, current_heatmap)
            smile_loss = torch.linalg.vector_norm(delta_latent)
            total_loss = (
                loss_parameters["lambda_landmark_loss"] * landmark_loss
                + loss_parameters["lambda_smile_loss"] * smile_loss
            )
            print(landmark_loss)
            total_loss.backward()
            optimizer.step()

            if i == self.num_iterations - 1:
                utils.save_image(
                    current_image,
                    f"sample/optimized_smile_image.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )

        return

    def optimize_retarget(self, source_latent: Tensor, target_latent: Tensor) -> Tensor:
        # Tensor (just a vector) that is added to the source latent to produce the retargeted image
        delta_latent = torch.zeros_like(source_latent)
        delta_latent.requires_grad = True

        # Instantiate optimizer (ADAM) as well as loss functions, optimize latent_delta
        optimizer = optim.Adam([delta_latent])
        mse_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()

        # Generate target image and compute its heatmap. Will remain fixed for the entire optmization process
        target_image = self.generate_from_sample_z(target_latent)
        target_heatmap = self.lip_detector.detect_lips(
            self.lip_detector.preprocess_image_from_tensor(target_image)
        ).detach()

        utils.save_image(
            target_image,
            f"sample/target.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

        # Create loss parameters
        loss_parameters = {}
        loss_parameters["lambda_landmark_loss"] = 5000.0
        loss_parameters["lambda_smoothness"] = 0.0  # Regularization term

        # Map the source z into w
        source_w = self.generator.style(source_latent).detach()

        for i in tqdm(range(self.num_iterations)):
            optimizer.zero_grad()
            # Generate image from latent code + current delta and compute its heatmap
            current_image = self.generate_from_sample_w(source_w + delta_latent)
            current_heatmap = self.lip_detector.detect_lips(
                self.lip_detector.preprocess_image_from_tensor(current_image)
            )

            # The mouth loss is the MSE between the heatmap of the current image to the heatmap of the target image
            mouth_loss = loss_parameters["lambda_landmark_loss"] * mse_loss(
                current_heatmap, target_heatmap
            )

            reg_loss = loss_parameters["lambda_smoothness"] * l1_loss(
                source_w, source_w + delta_latent
            )

            # Total loss is the sum of heatmap loss and the regularization term.
            # Compute loss and backpropagate
            total_loss = mouth_loss + reg_loss
            total_loss.backward()
            optimizer.step()

            if i % (self.num_iterations - 1) == 0:
                utils.save_image(
                    current_image,
                    f"sample/{str(i).zfill(6)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )

        return delta_latent
