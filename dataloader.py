import config
from dataset import SentimentDataset
from torch.utils.data import DataLoader

# create datasets from preprocessed train and test files and then return data loaders for both
def get_loader(rootPath, train_batch_size=32, test_batch_size = 8, shuffle=True, num_workers=8, pin_memory=True):
    trainDataset = SentimentDataset(rootPath + 'train.pkl', config.tokenizer, config.MAX_LEN)
    testDataset = SentimentDataset(rootPath + 'test.pkl', config.tokenizer, config.MAX_LEN)

    trainLoader = DataLoader(
        dataset=trainDataset,
        batch_size=train_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    testLoader = DataLoader(
        dataset=testDataset,
        batch_size=test_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return trainLoader, testLoader, config.tokenizer
