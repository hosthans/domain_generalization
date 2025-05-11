import pandas as pd
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

# The PACS dataset is available on Hugging Face and consists of a single 'train' split
pacs = pd.DataFrame(load_dataset("flwrlabs/pacs")['train'])


class PACSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, target_transform=None):
        self.dataframe = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image, _, label = self.dataframe.iloc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_data_loaders(
        target_domain: str,
        train_batch_size: int,
        test_batch_size=64,
        shuffle_train=True,
        shuffle_test=False,
        transform=None,
        target_transform=None) -> tuple[DataLoader, DataLoader]:
    """
    Returns a train and a test data loader for the PACS dataset for cross validation at a domain generalization task.

    Parameters:
        target_domain (str): The target domain for cross validation.
        train_batch_size (int): The batch size of the training set.
        test_batch_size (int=0): The batch size of the testing set. The whole testing set if set to 0.
        shuffle_train (bool=True): Whether to shuffle the train set.
        suffle_test (bool=False): Whether to shuffle the test set.
        transform: The transformation to apply to every image in the dataset.
        target_transform: The transformation to apply to the labels in the dataset.
    """
    domains = set(pacs['domain'])
    source_domains = domains - {target_domain}

    train_df = pacs[pacs['domain'].isin(source_domains)]
    test_df = pacs[pacs['domain'] == target_domain]

    train_dataset = PACSDataset(train_df, transform, target_transform)
    test_dataset = PACSDataset(test_df, transform, target_transform)

    if test_batch_size < 1:
        test_batch_size = len(test_dataset)

    train_loader = DataLoader(train_dataset, train_batch_size, shuffle_train)
    test_loader = DataLoader(test_dataset, test_batch_size, shuffle_test)

    return train_loader, test_loader


def get_nomalization_stats(target_domain):
    """
    Returns mean and standard deviation of the images in the train split for the given target domain.
    """
    train_df = pacs[pacs['domain'] != target_domain]
    transform = T.ToTensor()
    train_dataset = PACSDataset(train_df, transform)
    train_loader = DataLoader(train_dataset, batch_size=64)

    mean_sum = torch.zeros(3)
    var_sum = torch.zeros(3)
    total_pcount = 0

    for images, _ in train_loader:
        # images shape: (b, 3, h, w)
        batch_pcount = images.shape[0] * images.shape[2] * images.shape[3]
        batch_mean = images.mean(dim=(0, 2, 3))
        mean_sum += batch_mean * batch_pcount
        total_pcount += batch_pcount

    mean = mean_sum / total_pcount

    for images, _ in train_loader:
        # images shape: (b, 3, h, w)
        batch_pcount = images.shape[0] * images.shape[2] * images.shape[3]
        batch_var = ((images - mean.view(1, 3, 1, 1)) ** 2).mean(dim=(0, 2, 3))
        var_sum += batch_var * batch_pcount

    std = torch.sqrt(var_sum / total_pcount)

    return mean, std
