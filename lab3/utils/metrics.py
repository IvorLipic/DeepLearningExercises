from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def compute_metrics(logits, labels):
    preds = (logits > 0).float()
    labels = labels.float()
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    return acc, f1, cm