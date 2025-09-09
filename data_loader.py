import pickle
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        if train:
            for i in range(1, 6):
                batch = self._load_batch(os.path.join(data_dir, f'data_batch_{i}'))
                self.data.append(batch[b'data'])
                self.labels += batch[b'labels']
        else:
            batch = self._load_batch(os.path.join(data_dir, 'test_batch'))
            self.data.append(batch[b'data'])
            self.labels += batch[b'labels']

        self.data = np.concatenate(self.data).reshape(-1, 3, 32, 32).astype(np.uint8)

    def _load_batch(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='bytes')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx].transpose(1, 2, 0)  # CHW -> HWC
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_loaders(root_dir, batch_size=64):
    data_dir = os.path.join(root_dir, 'cifar-10-batches-py')

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    train_set = CIFAR10Dataset(data_dir, train=True, transform=train_transform)
    test_set = CIFAR10Dataset(data_dir, train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
