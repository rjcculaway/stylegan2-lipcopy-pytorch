# StyleGAN 2 LipCopy in PyTorch
This fork of StyleGAN 2 modifies the face's expression after generation using a predefined editing direction. It then incorporates elements from [StylePortraitVideo](https://style-portrait-video.github.io/)'s expression optimization process to ensure that the edited image's lip expression matches closely with the original's while maintaining the changed expression.

## Requirements
This fork works with the following configuration:
- PyTorch 1.3.1
- CUDA 10.1/10.2

## Usage

### Generate Samples

> python generate.py --sample N_FACES --pics N_PICS --ckpt PATH_CHECKPOINT

You should change your size (--size 256 for example) if you train with another dimension.

### Generate Samples and Perform Expression Optimization

> python generate_and_copy_lips.py --sample N_FACES --pics N_PICS --ckpt PATH_CHECKPOINT

You should change your size (--size 256 for example) if you train with another dimension.


## Samples
![Generated Image](https://github.com/rjcculaway/stylegan2-lipcopy-pytorch/assets/55573146/9b3a5dc8-d62f-481a-925a-9e45b7166b04)
![Generated Image + Editing](https://github.com/rjcculaway/stylegan2-lipcopy-pytorch/assets/55573146/eac6eebc-8408-47ef-b4c1-4961f98e4720)
![Optimized Expression](https://github.com/rjcculaway/stylegan2-lipcopy-pytorch/assets/55573146/2dbf26b2-7b2d-4341-a8f5-f5cdfd24c437)

## License

Model details and custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan2

Codes for Learned Perceptual Image Patch Similarity, LPIPS came from https://github.com/richzhang/PerceptualSimilarity

To match FID scores more closely to tensorflow official implementations, I have used FID Inception V3 implementations in https://github.com/mseitzer/pytorch-fid
