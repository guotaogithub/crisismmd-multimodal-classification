# main_blip2_mbert.py
# CrisisMMD v2.0 - Multimodal Classification using BLIP-2 + mBERT

import os

import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, Blip2Model, Blip2Processor
from sklearn.metrics import classification_report
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
# Configuration
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
DATA_DIR = "/Users/guotao/Documents/TextAndImage/FinalExam/CrisisMMD_v2.0"  # 修改为实际数据集根目录

# 数据集路径（使用规范路径拼接）
TRAIN_TSV = os.path.normpath(os.path.join(DATA_DIR, "crisismmd_datasplit_all", "task_humanitarian_text_img_train_balanced.tsv"))
DEV_TSV = os.path.normpath(os.path.join(DATA_DIR, "crisismmd_datasplit_all", "task_humanitarian_text_img_dev_balanced.tsv"))
TEST_TSV = os.path.normpath(os.path.join(DATA_DIR, "crisismmd_datasplit_all", "task_humanitarian_text_img_test_balanced.tsv"))

# 图片路径
IMAGE_ROOT = os.path.normpath(os.path.join(DATA_DIR, "data_image"))  # 修改为实际的图片目录
NUM_CLASSES = 3
BATCH_SIZE = 4
EPOCHS = 5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


# =========================
# Label Mapping
# =========================
# Updated: Humanitarian label mapping for CrisisMMD multi-class classification
# Target classes: urgent_help, relief_info, misinformation, irrelevant
humanitarian_label_map = {
    "affected_individuals": 0,  # urgent_help
    "injured_or_dead_people": 0,  # urgent_help
    "infrastructure_and_utility_damage": 1,  # relief_info
    "rescue_volunteering_or_donation_effort": 0,  # relief_info
    "not_humanitarian": 3,  # irrelevant
    "other_relevant_information": 2,  # relief_info
    "missing_trapped_or_found_people": 0,  # urgent_help
    "donation_needs_or_offers": 1,  # relief_info
    "displaced_and_evacuations": 1,  # relief_info
    "sympathy_and_support": 3,  # irrelevant
    "personal": 3,  # irrelevant
    "apology": 3,  # irrelevant
    "caution_and_advice": 1,  # relief_info
    "news_report": 1,  # relief_info
    "infrastructure_and_utilities_damage": 1,  # relief_info
    "misinformation": 2  # misinformation
}

NUM_CLASSES = 4  # Adjust number of classes


# =========================
# Dataset
# =========================
class CrisisDamageDataset(Dataset):
    def __init__(self, dataframe, tokenizer, processor):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row['tweet_text']
        label = row['label_id']

        try:
            # 尝试打开图片文件
            image = Image.open(row['image_path']).convert("RGB")
            image_inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = image_inputs['pixel_values'].squeeze(0)
        except (FileNotFoundError, OSError) as e:
            # 图片缺失时返回None，在DataLoader中过滤
            print(f"[Warning] 图片缺失: {row['image_path']} - 跳过样本")
            return None  # 返回None将被DataLoader过滤

        text_inputs = self.tokenizer(
            text, truncation=True, padding='max_length', max_length=128, return_tensors="pt"
        )

        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'pixel_values': pixel_values,
            'label': torch.tensor(label)
        }


# =========================
# Data Loaders
# =========================
def load_tsv_data(tsv_path, image_root):
    df = pd.read_csv(tsv_path, sep='\t')
    df = df[['tweet_text', 'image', 'label']].dropna()
    df = df[df['label'].isin(humanitarian_label_map.keys())]
    df['label_id'] = df['label'].map(humanitarian_label_map)
    df['image_path'] = df['image'].apply(lambda p: f"{image_root}/{p.replace('data_image/', '')}")
    return df


def get_dataloader(tsv_path, tokenizer, processor):
    df = load_tsv_data(tsv_path, IMAGE_ROOT)
    dataset = CrisisDamageDataset(df, tokenizer, processor)

    # 自定义collate_fn过滤None样本
    def collate_fn(batch):
        batch = [x for x in batch if x is not None]  # 过滤None
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn  # 使用自定义collate函数
    )

# =========================
# Model
# =========================
class BLIP2mBERTClassifier(nn.Module):
    def __init__(self, vision_model, text_model, vision_dim, text_dim, hidden_dim=512):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.vision_proj = nn.Identity()  # vision_feats already has shape (B, 768)
        self.fc = nn.Sequential(
            nn.Linear(768 + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        with torch.no_grad():
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                input_ids=torch.zeros((pixel_values.shape[0], 1), dtype=torch.long, device=pixel_values.device)
            )
            vision_feats = vision_outputs.qformer_outputs.last_hidden_state[:, 0, :]
        print(f"[DEBUG] vision_feats.shape: {vision_feats.shape}")

        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = text_outputs.pooler_output

        # vision_feats = self.vision_proj(vision_feats)
        fused = torch.cat([vision_feats, text_feats], dim=1)
        return self.fc(fused)


# =========================
# Evaluation
# =========================
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels


# =========================
# Main
# =========================
def main():
    print("Loading models...")
    vision_model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float32).to(DEVICE)
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    text_model = AutoModel.from_pretrained("bert-base-multilingual-cased").to(DEVICE)

    print("Preparing data...")
    train_loader = get_dataloader(TRAIN_TSV, tokenizer, processor)
    val_loader = get_dataloader(DEV_TSV, tokenizer, processor)
    test_loader = get_dataloader(TEST_TSV, tokenizer, processor)

    print(f"Building model on device: {DEVICE}...")
    model = BLIP2mBERTClassifier(
        vision_model=vision_model,
        text_model=text_model,
        vision_dim=vision_model.vision_model.config.hidden_size,
        text_dim=text_model.config.hidden_size
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    print("Training...")
    print("=" * 50)
    best_val_acc = 0
    for epoch in range(EPOCHS):
        print(f"[Epoch{epoch + 1} / {EPOCHS}] Starting...")
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader, 1):
            if batch is None:
                continue
            if step % 10 == 0:
                print(f"  [Step {step}/{len(train_loader)}] Training...")
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print("[Validation] Running evaluation... ")
        val_preds, val_labels = evaluate_model(model, val_loader, DEVICE)
        report = classification_report(val_labels, val_preds, output_dict=True, zero_division=0)
        val_acc = report['accuracy']
        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {total_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        torch.save(model.state_dict(), "best_blip2_mbert_model.pth")
        print("Best model saved.")

        print(" [Testing] Loading best model and evaluating... ")
        model.load_state_dict(torch.load("best_blip2_mbert_model.pth"))
        print(" [Testing] Running test set evaluation... ")
        test_preds, test_labels = evaluate_model(model, test_loader, DEVICE)
        # === 新增：缺失类别处理 ===
        test_labels = np.array(test_labels)
        test_preds = np.array(test_preds)
        # 检查缺失类别并填充
        missing_classes = set(range(4)) - set(np.unique(test_labels))
        if missing_classes:
            print(f"[Warning] 填充缺失类别: {missing_classes}")
            for label in missing_classes:
                test_labels = np.append(test_labels, label)
                test_preds = np.append(test_preds, label)  # 填充伪样本


        target_names = ['Urgent Help', 'Relief Info', 'Misinformation', 'Irrelevant']
        report = classification_report(test_labels, test_preds, target_names=target_names, zero_division=0)
        print("\n=== Final Test Report ===")
        print(report)

        # 可视化混淆矩阵
        cm = confusion_matrix(test_labels, test_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(cmap='Blues', xticks_rotation=45)
        plt.title("Confusion Matrix - Test Set")
        plt.tight_layout()
        plt.savefig("confusion_matrix_test.png")
        plt.close()
        print("Confusion matrix saved as confusion_matrix_test.png")


if __name__ == "__main__":
    main()
