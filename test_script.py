import torchvision
import torch

trainset = torchvision.datasets.MNIST(
    root='./data/',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torch.flatten]) # Scales Image values between [0, 255] to [0, 1]
)

