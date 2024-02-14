import torch 
import config
import pickle
import numpy as np
from sklearn import metrics

# Function to take true and predicted labels and calculate and print multiple metrics
def print_metrics(true, pred, loss, type):
    pred = np.array(pred) >= 0.35
    hamming_loss = metrics.hamming_loss(true,pred)
    precision_micro = metrics.precision_score(true, pred, average='micro',zero_division = 1)
    recall_micro = metrics.recall_score(true, pred, average='micro',zero_division = 1)
    precision_macro = metrics.precision_score(true, pred, average='macro',zero_division = 1)
    recall_macro = metrics.recall_score(true, pred, average='macro',zero_division = 1)
    f1_score_micro = metrics.f1_score(true, pred, average='micro',zero_division = 1)
    f1_score_macro = metrics.f1_score(true, pred, average='macro',zero_division = 1)
    print("-------{} Evaluation--------".format(type))
    print("BCE Loss: {:.4f}".format(loss))
    print("Hamming Loss: {:.4f}".format(hamming_loss))
    print("Precision Micro: {:.4f}, Recall Micro: {:.4f}, F1-measure Micro: {:.4f}".format(precision_micro, recall_micro, f1_score_micro))
    print("Precision Macro: {:.4f}, Recall Macro: {:.4f}, F1-measure Macro: {:.4f}".format(precision_macro, recall_macro, f1_score_macro))
    print("------------------------------------")
    return f1_score_micro, f1_score_macro, hamming_loss, loss 

# fucntion to save the metrics for model analysis 
def save_metrics(eval_metrics,file_name):
    eval = open('output/{}_metrics.pkl'.format(file_name), 'ab') 
    pickle.dump(eval_metrics, eval)                      
    eval.close()
    return True

# fucntion to save the model for inference
def save_checkpoint(state, filename=config.MODEL_PATH):
    print("=> Saving Model")
    torch.save(state, filename)