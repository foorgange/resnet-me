import torch
from sklearn.metrics import f1_score

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    acc = (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean().item()
    f1 = f1_score(all_targets, all_preds, average='weighted')
    return acc, f1
