import os   
import glob

import pandas as pd
import numpy as np

import os, glob


def load_unsup_texts(path):
    texts = []
    files = glob.glob(os.path.join(path, "*.txt"))  
    for f in files:
        with open(f, encoding="utf-8") as file:
            texts.append(file.read().strip())
    return texts

unsup_texts = load_unsup_texts("/Users/lakshyasmac/Desktop/IMDB/train/unsup")

import re

def preprocess_for_mlm(text):
    text = text.strip()
    
    
    text = re.sub(r"<br\s*/?>", " ", text)
    
    
    text = re.sub(r"\s+", " ", text)
    
    return text
unsup_texts = [preprocess_for_mlm(t) for t in unsup_texts]
np.random.seed(42)
np.random.shuffle(unsup_texts)  

from transformers import BertTokenizer, BertForMaskedLM

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128   
mlm_model = BertForMaskedLM.from_pretrained(MODEL_NAME)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_for_mlm(texts, tokenizer, max_len=128):
    return tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )

unsup_encodings = tokenize_for_mlm(unsup_texts, tokenizer)

import torch
from torch.utils.data import Dataset

class MLMDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

    def __len__(self):
        return self.encodings["input_ids"].size(0)

mlm_dataset = MLMDataset(unsup_encodings)

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

mlm_training_args = TrainingArguments(
    output_dir="./bert_imdb_mlm",
    overwrite_output_dir=True,
    num_train_epochs=2,            
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    logging_steps=500,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True if torch.cuda.is_available() else False
)
mlm_trainer = Trainer(
    model=mlm_model,
    args=mlm_training_args,
    train_dataset=mlm_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

mlm_trainer.train()
mlm_model.save_pretrained("/Users/lakshyasmac/Desktop/IMDB/models/bert_imdb_domain_mlm")
tokenizer.save_pretrained("/Users/lakshyasmac/Desktop/IMDB/models/bert_imdb_domain_mlm")

def tokenize_for_cls(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

train_texts = []
train_labels = []

for label in ["pos", "neg"]:
    path = f"/Users/lakshyasmac/Desktop/IMDB/train/{label}/*.txt"
    files = glob.glob(path)
    for f in files:
        with open(f, encoding="utf-8") as file:
            text = file.read().strip()
            text = preprocess_for_mlm(text)
            train_texts.append(text)
            train_labels.append(1 if label == "pos" else 0)

test_texts = []
test_labels = []

for label in ["pos", "neg"]:
    path = f"/Users/lakshyasmac/Desktop/IMDB/test/{label}/*.txt"
    files = glob.glob(path)
    for f in files:
        with open(f, encoding="utf-8") as file:
            text = file.read().strip()
            text = preprocess_for_mlm(text)
            test_texts.append(text)
            test_labels.append(1 if label == "pos" else 0)



from transformers import BertTokenizer, BertForSequenceClassification

mlm_model_path = "/Users/lakshyasmac/Desktop/IMDB/models/bert_imdb_domain_mlm"

tokenizer = BertTokenizer.from_pretrained(mlm_model_path)

model = BertForSequenceClassification.from_pretrained(
    mlm_model_path,
    num_labels=2
)


X_train = tokenize_for_cls(train_texts)
X_test = tokenize_for_cls(test_texts)


import torch
from torch.utils.data import Dataset

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


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./logs/sentiment",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs/sentiment"
)


import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs[:, 1])
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("/Users/lakshyasmac/Desktop/IMDB/models/bert_imdb_domain_cls")

import matplotlib.pyplot as plt

train_loss = trainer.state.log_history
train_epochs = []
train_losses = []
val_losses = []

for log in train_loss:
    if "loss" in log and "epoch" in log:
        train_epochs.append(log["epoch"])
        train_losses.append(log["loss"])
    if "eval_loss" in log:
        val_losses.append(log["eval_loss"])

plt.figure()
plt.plot(train_losses)
plt.plot(val_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend(["Train Loss", "Validation Loss"])
plt.show()
from sklearn.metrics import roc_curve, auc
import numpy as np

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
plt.title("ROC Curve")
plt.legend()
plt.show()
from sklearn.metrics import confusion_matrix
import seaborn as sns

pred_labels = np.argmax(logits, axis=1)
cm = confusion_matrix(labels, pred_labels)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
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
