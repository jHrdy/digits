import torchvision
import torch 

data = torchvision.datasets.MNIST(
    root='./data/',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torch.flatten])
)