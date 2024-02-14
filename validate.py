import config
import torch
from utils import print_metrics
device = config.device

def loss_fun(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

# function to validate the validation data from trained model
def validate(model, testLoader):
    model.eval()
    val_targets = []
    val_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testLoader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            loss = loss_fun(outputs, targets)
            epoch_loss = loss.item()
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

        return print_metrics(val_targets,val_outputs, epoch_loss,'Validation')