import torch
import numpy as np
import pickle
import config
from transformers import BertTokenizer, BertConfig
from model import SentimentMultilabel,SentimentMultilabelLarge
from dataloader import get_loader
from validate import validate
from utils import save_checkpoint,print_metrics, save_metrics                                 

device = config.device
model_config = BertConfig()
num_labels = config.NUM_LABELS
lr = config.LEARNING_RATE
epochs = config.EPOCHS

# evaluation metrics data, for analysis of models performance
eval_metrics = {
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "training_f1_micro": [],
            "training_f1_macro": [],
            "val_f1_micro": [],
            "val_f1_macro": [],
            "training_hamming_loss": [],
            "val_hamming_loss": [],
        }

# Binary cross entropy with logits loss function
def loss_fun(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

# function to train the model 
def train():
    model = SentimentMultilabel(num_labels,model_config).to(device) if config.model == "base" else SentimentMultilabelLarge(num_labels,model_config).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    # creating the training and validation data loaders 
    trainLoader, testLoader, _ = get_loader('output/')

    for epoch in range(1,epochs+1):
        eval_metrics["epochs"].append(epoch)
        model.train()
        epoch_loss = 0
        # training actual and prediction for each epoch for printing metrics
        train_targets = []
        train_outputs = []
        for _, data in enumerate(trainLoader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fun(outputs, targets)
            epoch_loss = loss.item()
            train_targets.extend(targets.cpu().detach().numpy().tolist())
            train_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            if _ % 50 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # calculating the evaluation scores for both training and validation data
        train_f1_micro, train_f1_macro, train_hamming,train_loss = print_metrics(train_targets,train_outputs,epoch_loss, 'Training')
        val_f1_micro, val_f1_macro, val_hamming, val_loss = validate(model, testLoader)
        eval_metrics['training_f1_micro'].append(train_f1_micro)
        eval_metrics['training_f1_macro'].append(train_f1_macro)
        eval_metrics['training_hamming_loss'].append(train_hamming)
        eval_metrics['val_f1_micro'].append(val_f1_micro)
        eval_metrics['val_f1_macro'].append(val_f1_macro)
        eval_metrics['val_hamming_loss'].append(val_hamming)
        eval_metrics["train_loss"].append(train_loss)
        eval_metrics["val_loss"].append(val_loss)
    
    # saving the metrics and trained model for inference and model analysis
    save_metrics(eval_metrics,'bert_base' if config.model == 'base' else 'bert_large')
    checkpoint = {"state_dict": model.state_dict()}
    save_checkpoint(checkpoint)
    return True

if __name__ == "__main__":
    train()
    