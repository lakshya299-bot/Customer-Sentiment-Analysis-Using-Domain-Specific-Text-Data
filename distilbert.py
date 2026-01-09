import os
import glob
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)




def preprocess_text(text):
    text = text.strip()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text





train_texts, train_labels = [], []
test_texts, test_labels = [], []

for label in ["pos", "neg"]:
    label_value = 1 if label == "pos" else 0

    train_files = glob.glob(f"/Users/lakshyasmac/Desktop/IMDB/train/{label}/*.txt")
    for f in train_files:
        with open(f, encoding="utf-8") as file:
            train_texts.append(preprocess_text(file.read()))
            train_labels.append(label_value)

    test_files = glob.glob(f"/Users/lakshyasmac/Desktop/IMDB/test/{label}/*.txt")
    for f in test_files:
        with open(f, encoding="utf-8") as file:
            test_texts.append(preprocess_text(file.read()))
            test_labels.append(label_value)





DISTILBERT_MODEL = "distilbert-base-uncased"

tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL)

model = DistilBertForSequenceClassification.from_pretrained(
    DISTILBERT_MODEL,
    num_labels=2
)





def tokenize_for_cls(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

X_train = tokenize_for_cls(train_texts)
X_test = tokenize_for_cls(test_texts)





class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SentimentDataset(X_train, train_labels)
test_dataset = SentimentDataset(X_test, test_labels)





def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs[:, 1])
    }





training_args = TrainingArguments(
    output_dir="./logs/distilbert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs/distilbert"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("/Users/lakshyasmac/Desktop/IMDB/models/distilbert_imdb_cls")
tokenizer.save_pretrained("/Users/lakshyasmac/Desktop/IMDB/models/distilbert_imdb_cls")





log_history = trainer.state.log_history
train_losses, val_losses = [], []

for log in log_history:
    if "loss" in log:
        train_losses.append(log["loss"])
    if "eval_loss" in log:
        val_losses.append(log["eval_loss"])

plt.figure()
plt.plot(train_losses)
plt.plot(val_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DistilBERT: Training vs Validation Loss")
plt.legend(["Train Loss", "Validation Loss"])
plt.show()





predictions = trainer.predict(test_dataset)
logits = predictions.predictions
labels = predictions.label_ids

probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
fpr, tpr, _ = roc_curve(labels, probs[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("DistilBERT ROC Curve")
plt.legend()
plt.show()





pred_labels = np.argmax(logits, axis=1)
cm = confusion_matrix(labels, pred_labels)

plt.figure()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("DistilBERT Confusion Matrix")
plt.show()

import numpy as np
pred_labels = np.argmax(logits, axis=1)
cm = confusion_matrix(labels, pred_labels)
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

precision = precision_score(labels, pred_labels)
recall = recall_score(labels, pred_labels)
f1 = f1_score(labels, pred_labels)

print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1-score  :", round(f1, 4))
print("\nClassification Report:\n", classification_report(labels, pred_labels, target_names=["Negative", "Positive"]))
