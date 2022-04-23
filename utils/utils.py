import torch
import numpy as np
import random

def set_seed(manualSeed):
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    CUDA = True if torch.cuda.is_available() else False
    if CUDA:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(manualSeed)

def masked_loss(output, labels, mask, loss_f=torch.nn.CrossEntropyLoss()):
    return loss_f(output[mask], labels[mask])

def get_accuracy(output, labels):
    if len(labels.shape) == 2:
        labels = torch.argmax(labels, dim=1)
    pred_y = torch.max(output, dim=1)[1]
    correct = torch.sum(pred_y == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate(output, labels, train_mask, val_mask, test_mask):
    train_acc = get_accuracy(output[train_mask], labels[train_mask])
    val_acc = get_accuracy(output[val_mask], labels[val_mask])
    test_acc = get_accuracy(output[test_mask], labels[test_mask])
    return train_acc, val_acc, test_acc


