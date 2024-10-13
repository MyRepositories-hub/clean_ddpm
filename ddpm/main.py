import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dit import DiT


def get_args():
    """
    Hyperparameter settings for DDPM
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--t_max', type=int, default=1000)
    parser.add_argument('--sigma', type=float, default=0.02)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--update_epoch', type=int, default=80)
    parser.add_argument('--use_cuda', type=bool, default=True)
    return parser.parse_args()


def load_mnist(batch_size):
    """
    Load MNIST data and normalize it to the range -1 to 1 to match the Gaussian distribution
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def generate_forward_samples(batch_init_image, batch_t, alpha_bar):
    """
    Obtain the images in the batch after adding noise for a random number of steps to each image
    """
    device = batch_init_image.device
    batch_epsilon_t = torch.randn_like(batch_init_image).to(device)
    batch_alpha_bar = alpha_bar[batch_t].view(-1, 1, 1, 1).to(device)
    batch_image_t = torch.sqrt(batch_alpha_bar) * batch_init_image + torch.sqrt(1 - batch_alpha_bar) * batch_epsilon_t
    return batch_image_t, batch_epsilon_t


def train():
    """
    The training process of DDPM
    """
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    model = DiT(img_size=28, patch_size=4, channel=1, emb_size=64, label_num=10, dit_num=3, head=4).to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    alpha = 1 - torch.linspace(args.beta_start, args.beta_end, args.t_max).to(device)
    alpha_bar = torch.cumsum(alpha.log(), dim=0).exp().to(device)

    pbar = tqdm(range(args.update_epoch))
    for _ in pbar:
        pbar.set_description('Training loss')
        for image, label in load_mnist(args.batch_size):
            if len(label) != args.batch_size:
                continue

            image, label = image.to(device), label.to(device)
            batch_t = torch.randint(0, args.t_max, size=(args.batch_size,)).to(device)
            batch_image_t, batch_epsilon_t = generate_forward_samples(image, batch_t, alpha_bar)
            batch_epsilon_pre = model(batch_image_t, batch_t, label)

            loss = loss_function(batch_epsilon_t, batch_epsilon_pre)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), './model.pth')


if __name__ == '__main__':
    train()
