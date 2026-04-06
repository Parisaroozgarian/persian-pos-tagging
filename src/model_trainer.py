import torch
from torch.optim import AdamW  # ✅ NOT from transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
import streamlit as st
import numpy as np


class PersianPOSDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        tags = item["upos"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = encoding.word_ids()
        labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word_id:
                tag = tags[word_id] if word_id < len(tags) else "NOUN"
                labels.append(self.label2id.get(tag, 0))
            else:
                labels.append(-100)
            prev_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class ModelTrainer:
    def __init__(self, model_name="distilbert-base-multilingual-cased"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ✅ auto-detect
        self.tokenizer = None
        self.model = None

    def setup_model(self, label2id, id2label):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)

    def apply_freezing_strategy(self, strategy):
        """Freeze layers based on strategy name."""
        if strategy == "no_freezing":
            for param in self.model.parameters():
                param.requires_grad = True

        elif strategy == "freeze_embeddings":
            for param in self.model.distilbert.embeddings.parameters():
                param.requires_grad = False

        elif strategy == "freeze_first_3":
            for param in self.model.distilbert.embeddings.parameters():
                param.requires_grad = False
            for i, layer in enumerate(self.model.distilbert.transformer.layer):
                if i < 3:
                    for param in layer.parameters():
                        param.requires_grad = False

        elif strategy == "freeze_all_but_classifier":
            for param in self.model.distilbert.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def train(
        self,
        train_data,
        val_data,
        label2id,
        id2label,
        freezing_strategy="no_freezing",
        epochs=1,
        batch_size=32,
        learning_rate=2e-5,
        progress_callback=None,
    ):
        self.setup_model(label2id, id2label)
        self.apply_freezing_strategy(freezing_strategy)

        train_dataset = PersianPOSDataset(train_data, self.tokenizer, label2id)
        val_dataset = PersianPOSDataset(val_data, self.tokenizer, label2id)

        # ✅ num_workers=0 — required for Streamlit Cloud (no multiprocessing)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
        )

        results = []

        for epoch in range(epochs):
            # --- Training ---
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            # ✅ Plain enumerate — no tqdm
            for i, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                preds = outputs.logits.argmax(dim=-1)
                mask = labels != -100
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()

                # Report progress every 10 batches
                if progress_callback and i % 10 == 0:
                    progress_callback(
                        epoch=epoch + 1,
                        batch=i,
                        total_batches=len(train_loader),
                        loss=total_loss / (i + 1),
                        accuracy=correct / total if total > 0 else 0,
                    )

            train_loss = total_loss / len(train_loader)
            train_acc = correct / total if total > 0 else 0

            # --- Validation ---
            val_loss, val_acc = self._evaluate(val_loader)

            results.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "freezing_strategy": freezing_strategy,
            })

        return results, self.model, self.tokenizer

    def _evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            # ✅ Plain enumerate — no tqdm
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()

                preds = outputs.logits.argmax(dim=-1)
                mask = labels != -100
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()

        return total_loss / len(loader), correct / total if total > 0 else 0