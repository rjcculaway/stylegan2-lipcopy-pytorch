import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

from detect_lips import LipDetector

def generate(args, g_ema, device, mean_latent):
  with torch.no_grad():
      g_ema.eval()
      sample_z = torch.randn(args.sample, args.latent, device=device)

      sample, _ = g_ema(
          [sample_z], truncation=args.truncation, truncation_latent=mean_latent
      )

      return sample
    
def generate_from_sample(args, g_ema: Generator, device, mean_latent, sample_z):
  with torch.no_grad():
    g_ema.eval()
    sample, _ = g_ema(
       [sample_z.to(device)], truncation=args.truncation, truncation_latent=mean_latent
    )

    return sample


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    # Load lip detection module
    lip_detector = LipDetector()

    for i in tqdm(range(args.pics)):
      sample = generate(args, g_ema, device, mean_latent)
      grid = utils.make_grid(sample, nrow=1, normalize=True, value_range=(-1, 1)) * 2. - 1.
      # print(grid.size())
      # print(f"min: {grid.min()} max:  {grid.max()}")
      
      marks, heatmap = lip_detector.detect_lips(lip_detector.preprocess_image_from_tensor(grid))
      img = grid.permute(1, 2, 0).cpu().numpy()
      lip_detector.save_image_with_marks(img, marks, heatmap, f"{str(i).zfill(6)}")
