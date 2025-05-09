import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

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
        target_domain,
        train_batch_size,
        test_batch_size=128,
        shuffle_train=True,
        shuffle_test=False,
        transform=None,
        target_transform=None) -> tuple[DataLoader, DataLoader]:
    """
    Returns a train and a test data loader for the PACS dataset for cross validation at a domain generalization task.
    """
    domains = set(pacs['domain'])
    source_domains = domains - {target_domain}

    train_df = pacs[pacs['domain'].isin(source_domains)]
    test_df = pacs[pacs['domain'] == target_domain]

    train_dataset = PACSDataset(train_df, transform, target_transform)
    test_dataset = PACSDataset(test_df, transform, target_transform)

    train_loader = DataLoader(train_dataset, train_batch_size, shuffle_train)
    test_loader = DataLoader(test_dataset, test_batch_size, shuffle_test)

    return train_loader, test_loader
