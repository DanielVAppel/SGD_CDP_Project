from torchvision import datasets, transforms
from torch.utils.data import random_split

def get_mnist(root='./data', train_ratio=0.9):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    test  = datasets.MNIST(root=root, train=False, download=True, transform=tfm)
    n_train = int(len(train)*train_ratio)
    n_val = len(train) - n_train
    train_ds, val_ds = random_split(train, [n_train, n_val])
    return train_ds, val_ds, test
