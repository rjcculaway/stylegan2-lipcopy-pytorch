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

![Generated Image](https://github.com/rjcculaway/stylegan2-lipcopy-pytorch/assets/55573146/e68a2267-fdf1-4f73-ac02-40f4938a481e)
***Generated Image***
![Edited Image](https://github.com/rjcculaway/stylegan2-lipcopy-pytorch/assets/55573146/326ed721-96ae-4577-b6bb-56e1a41de508)
***Edited Image***
![Optimized Image](https://github.com/rjcculaway/stylegan2-lipcopy-pytorch/assets/55573146/ccea7493-f60e-448d-b0d0-1c78c8a90d1c)
***Optimized Image***

## License

Model details and custom CUDA kernel codes are from official repostiories: https://github.com/NVlabs/stylegan2

Codes for Learned Perceptual Image Patch Similarity, LPIPS came from https://github.com/richzhang/PerceptualSimilarity

To match FID scores more closely to tensorflow official implementations, I have used FID Inception V3 implementations in https://github.com/mseitzer/pytorch-fid
