import torch
import pandas as pd
from torch.utils.data import Dataset

# Pytorch dataset for the training of BERT model
class SentimentDataset(Dataset):
    def __init__(self, df_path, tokenizer, max_len):
        self.df = pd.read_pickle(df_path)
        self.texts = self.df.text 
        self.targets = self.df.encoded
        self.max_len = max_len 
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self,index):
        text = self.texts[index]
        target = self.targets[index]
        # encoding the texts with pretrained bert tokenizer
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.long)
        }
