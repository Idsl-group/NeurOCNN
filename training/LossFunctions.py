import torch

def accuracy_from_logits(logits, y_true):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == y_true).sum().item()
    total = y_true.numel()
    return correct / total