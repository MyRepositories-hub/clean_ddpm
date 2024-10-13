import os
import shutil

import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm

from dit import DiT
from main import get_args


def denormalize(tensor):
    """
    Denormalize image pixel values back to the range 0 to 255
    """
    tensor = (tensor + 1) / 2
    tensor = tensor * 255.0
    return tensor.clamp(0, 255).byte()


def plot(tensor, t):
    """
    Save the images from each step of the denoising process
    """
    tensor = denormalize(tensor).squeeze(1)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(tensor[i], vmin=0, vmax=255)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('./denoising_process/' + str(t) + '.png', bbox_inches='tight', pad_inches=0.15)
    plt.close()


def create_file(name):
    """
    Create a folder for the denoising process
    """
    if os.path.exists(name):
        if os.path.isdir(name):
            shutil.rmtree(name)
            os.makedirs(name)
        else:
            raise NotADirectoryError(f'{name} exists and is not a directory.')
    else:
        os.makedirs(name)


def create_gif(image_folder, output_gif, duration, t_max):
    """
    Create a gif file for the denoising process
    """
    frames = [Image.open(os.path.join(image_folder, str(t) + '.png')) for t in reversed(range(0, t_max, 20))]
    frames[0].save(
        output_gif,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0
    )


def inference():
    """
    The inference process of DDPM
    """
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    alpha = 1 - torch.linspace(args.beta_start, args.beta_end, args.t_max).to(device)
    alpha_bar = torch.cumsum(alpha.log(), dim=0).exp().to(device)

    model = DiT(img_size=28, patch_size=4, channel=1, emb_size=64, label_num=10, dit_num=3, head=4).to(device)
    model.load_state_dict(torch.load('./model.pth'))
    tensor = torch.randn(10, 1, 28, 28).to(device)
    label = torch.arange(10).to(device)
    # label = 3 * torch.ones(size=(10,)).long().to(device)  # you can also choose a specific number from 0 to 9
    create_file('./denoising_process')

    model.eval()
    pbar = tqdm(reversed(range(args.t_max)))
    for step in pbar:
        pbar.set_description('Denoising steps')
        t = torch.Tensor([step] * 10).int().to(device)
        tensor -= (1 - alpha[step]) / (torch.sqrt(1 - alpha_bar[step])) * model(tensor, t, label)
        tensor /= torch.sqrt(alpha[step])
        z = torch.randn_like(tensor) if step > 0 else 0.0
        tensor += args.sigma * z
        tensor = torch.clamp(tensor, -1, 1)
        plot(tensor.cpu(), step)
        pbar.set_postfix(t=step)

    create_gif('denoising_process', 'enjoy.gif', 30, args.t_max)


if __name__ == '__main__':
    inference()
