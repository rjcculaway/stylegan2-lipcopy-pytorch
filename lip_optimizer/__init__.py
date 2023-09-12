from numpy import ndarray
import numpy as np
import torch
from torch import optim, nn, Tensor
from torchvision import utils


# LipOptimizer uses landmark loss to guide optimization
class LipOptimizer:
    def __init__(self, generator, lip_detector, generator_args):
        self.DEVICE = "cuda:0"
        self.generator = generator
        self.lip_detector = lip_detector
        self.generator_args = generator_args
        self.num_iterations = 300
        self.parameters = []
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.l1_loss = nn.L1Loss()

        # Formula to get the new input latent code
        # input_ws = original_ws + (
        #         optim_target + direction_ws
        #     ) * tensor_edit_scale.view(bs, 1, 1)
        return

    # Given a latent code, generates an image.
    def generate_from_sample(self, sample_z):
        self.generator.eval()
        sample, _ = self.generator(
            [sample_z.to("cuda")],
            truncation=self.generator_args.truncation,
            truncation_latent=None,
        )

        return self.transform_sample(sample)

    # Maps the image to -1.0 to 1.0
    def transform_sample(self, sample: Tensor):
        return (
            utils.make_grid(sample, nrow=1, normalize=True, value_range=(-1, 1)) * 2.0
            - 1.0
        ).requires_grad_(True)

    def optimize(self, source_latent: Tensor, target_latent: Tensor):
        target_image = self.generate_from_sample(target_latent)
        target_heatmap = self.lip_detector.detect_lips(
            self.lip_detector.preprocess_image_from_tensor(target_image)
        )
        target_heatmap = torch.Tensor(target_heatmap)
        output_heatmap = torch.zeros_like(target_heatmap).to(self.DEVICE)

        lambda_mouth_distance = torch.tensor(1000.0).to(self.DEVICE)
        optimal_delta: Tensor = torch.zeros_like(source_latent).to(self.DEVICE)
        optimal_delta.requires_grad = True
        lambda_regularization: Tensor = torch.tensor(2000.0).to(self.DEVICE)
        total_loss = torch.tensor(1.0).to(self.DEVICE)

        self.optimizer = optim.Adam(
            [
                optimal_delta,
            ],
            lr=3e-4,
        )

        utils.save_image(
            target_image,
            f"sample/target.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

        for i in range(self.num_iterations):
            computed_latent = source_latent + optimal_delta
            output = self.generate_from_sample(computed_latent)

            output_heatmap = self.lip_detector.detect_lips(
                self.lip_detector.preprocess_image_from_tensor(output)
            )
            print(output_heatmap.grad)

            mouth_loss = (
                lambda_mouth_distance
                * self.mse_loss(output_heatmap, target_heatmap).sum()
            )
            regularization_loss = lambda_regularization * self.l1_loss(
                source_latent, source_latent + optimal_delta
            )

            print(f"{mouth_loss}")

            total_loss = mouth_loss + regularization_loss
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if i % (self.num_iterations - 1) == 0:
                utils.save_image(
                    output,
                    f"sample/{str(i).zfill(6)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                # self.lip_detector.save_image_with_marks(
                #     output.cpu().permute(1, 2, 0).numpy(),
                #     np.array(output_heatmap.unsqueeze(0)),
                #     heat,
                #     name_index=str(i).zfill(6),
                # )
