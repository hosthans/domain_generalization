import pandas as pd
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

# The Office-Home dataset is available on Hugging Face and consists of a single 'train' split
office_home = load_dataset("flwrlabs/office-home", split='train')


class OfficeHomeDataset(Dataset):
    def __init__(self, hf_dataset, indices=None, transform=None, target_transform=None):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.indices = indices if indices is not None else list(
            range(len(hf_dataset)))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        item = self.hf_dataset[idx]

        image, domain, label = item['image'], item['domain'], item['label']

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, domain, label


def get_domain_indices(dataset, target_domain):
    """
    Get indices for train and test splits
    """
    train_indices = []
    test_indices = []

    for i in range(len(dataset)):
        domain = dataset[i]['domain']
        if domain == target_domain:
            test_indices.append(i)
        else:
            train_indices.append(i)

    return train_indices, test_indices


def get_data_loaders(
        target_domain: str,
        train_batch_size: int,
        test_batch_size=16,
        shuffle_train=True,
        shuffle_test=False,
        transform=None,
        target_transform=None,
        drop_last=False) -> tuple[DataLoader, DataLoader]:
    """
    Returns a train and a test data loader for the Office-Home dataset for cross validation at a domain generalization task.

    Parameters:
        target_domain (str): The target domain for cross validation.
        train_batch_size (int): The batch size of the training set.
        test_batch_size (int=0): The batch size of the testing set. The whole testing set if set to 0.
        shuffle_train (bool=True): Whether to shuffle the train set.
        suffle_test (bool=False): Whether to shuffle the test set.
        transform: The transformation to apply to every image in the dataset.
        target_transform: The transformation to apply to the labels in the dataset.
        drop_last (bool=False): Whether to drop the last batch if it is smaller than the batch size.
    """
    train_indices, test_indices = get_domain_indices(
        office_home, target_domain)

    train_dataset = OfficeHomeDataset(
        office_home, train_indices, transform, target_transform)
    test_dataset = OfficeHomeDataset(
        office_home, test_indices, transform, target_transform)

    if test_batch_size < 1:
        test_batch_size = len(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        train_batch_size,
        shuffle_train,
        drop_last=drop_last
    )
    test_loader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle_test,
        drop_last=drop_last
    )

    return train_loader, test_loader


def get_normalization_stats(target_domain):
    """
    Returns the pre-computed mean and standard deviation of the images in the train split for the given target domain.
    """
    match target_domain:
        case 'Art':
            mean = torch.Tensor([0.6514, 0.6297, 0.6074])
            std = torch.Tensor([0.2099, 0.2110, 0.2179])
        case 'Clipart':
            mean = torch.Tensor([0.6426, 0.6185, 0.5941])
            std = torch.Tensor([0.1947, 0.1972, 0.2041])
        case 'Product':
            mean = torch.Tensor([0.5836, 0.5559, 0.5248])
            std = torch.Tensor([0.2086, 0.2082, 0.2144])
        case 'Real World':
            mean = torch.Tensor([0.6393, 0.6190, 0.5975])
            std = torch.Tensor([0.2169, 0.2172, 0.2230])
        case _:
            mean = torch.zeros(3)
            std = torch.zeros(3)
    return mean, std


def get_normalization_stats_(target_domain):
    """
    Returns mean and standard deviation of the images in the train split for the given target domain.
    Method taken from https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/31
    """
    train_indices, _ = get_domain_indices(office_home, target_domain)
    transform = T.Compose([
        T.Resize(256),
        T.ToTensor()
    ])
    dataset = OfficeHomeDataset(office_home, train_indices, transform)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    mean = 0.
    for images, _ in loader:
        # (b, 3, h, w)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.
    pixel_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0, 2])
        pixel_count += images.nelement()
    std = torch.sqrt(var / pixel_count)

    return mean, std
