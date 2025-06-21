import os
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import Blip2Processor, Blip2Model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tqdm import tqdm
from accelerate import Accelerator
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ========= 检查设备类型 =========
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# ========= 初始化 accelerator =========
accelerator = Accelerator()
device = accelerator.device
print(f"Accelerator device: {device}")

# ========= 加载模型和处理器 =========
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco",use_fast=True)

# 根据设备类型调整模型加载参数
if torch.backends.mps.is_available():
    # 苹果M4 Max环境
    blip2_model = Blip2Model.from_pretrained(
        "Salesforce/blip2-opt-2.7b-coco",
        torch_dtype=torch.float32,  # MPS不支持float16，使用float32
        device_map=None  # 不使用device_map，让accelerator处理
    )
else:
    # 其他环境保持原有设置
    blip2_model = Blip2Model.from_pretrained(
        "Salesforce/blip2-opt-2.7b-coco",
        torch_dtype=torch.float16,
        device_map="auto"
    )

# 启用梯度检查点以节省内存
blip2_model.gradient_checkpointing_enable()
# 禁用缓存以兼容梯度检查点
blip2_model.config.use_cache = False
blip2_model.eval()

# 冻结BLIP-2模型的所有参数
for param in blip2_model.parameters():
    param.requires_grad = False
                                                        
# ========= 数据路径 =========
# 使用相对路径，指向项目根目录下的dataset文件夹
base_path = "dataset/"
train_path = os.path.join(base_path, "train_balanced_100_interleaved.csv")
test_path = os.path.join(base_path, "test_balanced_100.csv")

# ========= 读取数据 =========
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
# 图像路径在CSV中已经是正确的相对路径，无需再次拼接
# train_df['image_path'] = train_df['image_path'].apply(lambda x: os.path.join(base_path, x))
# test_df['image_path'] = test_df['image_path'].apply(lambda x: os.path.join(base_path, x))

# ========= 标签映射 =========
label2id = {label: i for i, label in enumerate(sorted(train_df['label'].unique()))}
id2label = {v: k for k, v in label2id.items()}
train_df['label_id'] = train_df['label'].map(label2id)
test_df['label_id'] = test_df['label'].map(label2id)


# ========= Dataset 类 =========
class CrisisDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row['image_path']).convert("RGB")
        except:
            image = Image.new('RGB', (224, 224), (255, 255, 255))
        text = row['text']
        inputs = self.processor(images=image, text=text, return_tensors="pt", max_length=128, truncation=True,
                                padding="max_length")
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(row['label_id'], dtype=torch.long)
        }


# ========= 创建 DataLoader =========
train_dataset = CrisisDataset(train_df, processor)
test_dataset = CrisisDataset(test_df, processor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# ========= 计算 class_weight =========
class_weights = compute_class_weight(class_weight="balanced", classes=np.array(list(range(len(label2id)))),
                                     y=train_df['label_id'])
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("类别权重:", {id2label[i]: class_weights[i].item() for i in range(len(label2id))})

# 如果数据是平衡的，使用手动调整的权重来鼓励多样性
if torch.allclose(class_weights, torch.ones_like(class_weights)):
    print("数据是平衡的，使用手动调整的权重")
    # 重新平衡权重，提升irrelevant和relief_info
    manual_weights = torch.tensor([1.7, 1.5, 1.4, 1.6], dtype=torch.float32).to(device)
    class_weights = manual_weights

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ========= 模型定义 =========
class MultimodalClassifier(nn.Module):
    def __init__(self, blip2_model, num_classes):
        super().__init__()
        self.blip2 = blip2_model
        hidden_size = self.blip2.vision_model.config.hidden_size + self.blip2.qformer.config.hidden_size
        
        # 简化的分类器架构，避免过拟合
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                # 使用较小的初始化权重
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, pixel_values, input_ids, attention_mask):
        # 获取BLIP-2的输出
        outputs = self.blip2(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 简单的特征融合
        vision_embeds = outputs.vision_outputs.last_hidden_state[:, 0, :]
        text_embeds = outputs.qformer_outputs.last_hidden_state[:, 0, :]
        fused = torch.cat([vision_embeds, text_embeds], dim=1)
        
        # 确保fused tensor在正确的设备上
        fused = fused.to(self.classifier[0].weight.device)
        return self.classifier(fused)

# ========= 初始化模型、损失、优化器 =========
model = MultimodalClassifier(blip2_model, num_classes=len(label2id))

# 使用更简单的优化器设置
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.02)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)

# ========= 使用 Accelerator 准备组件 =========
model, optimizer, train_loader, test_loader, criterion = accelerator.prepare(
    model, optimizer, train_loader, test_loader, criterion
)
# 单独准备scheduler
scheduler = accelerator.prepare(scheduler)


# ========= 训练函数 =========
def train_one_epoch(model, dataloader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader):
        gc.collect()
        # 根据设备类型清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            # MPS环境下的内存清理
            torch.mps.empty_cache()
        
        with accelerator.accumulate(model):
            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # 使用带权重的CrossEntropyLoss
            labels = batch['label'].to(outputs.device)
            loss = criterion(outputs, labels)
            
            # 计算准确率
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    return avg_loss


# ========= 评估函数 =========
def evaluate(model, dataloader):
    model.eval()
    preds, targets = [], []
    total_loss = 0
    correct = 0
    total = 0
    
    # 用于统计每个类别的预测情况
    class_correct = {i: 0 for i in range(len(label2id))}
    class_total = {i: 0 for i in range(len(label2id))}
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # 计算损失和准确率
            labels = batch['label'].to(outputs.device)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # 统计每个类别的准确率
            for i in range(len(label2id)):
                mask = (labels == i)
                class_total[i] += mask.sum().item()
                class_correct[i] += (pred[mask] == labels[mask]).sum().item()
            
            pred = accelerator.gather_for_metrics(outputs.argmax(dim=1)).cpu().numpy()
            target = accelerator.gather_for_metrics(batch['label']).cpu().numpy()
            preds.extend(pred)
            targets.extend(target)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # 显示每个类别的准确率
    print("\n各类别准确率:")
    for i in range(len(label2id)):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f"{id2label[i]}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
    
    # 使用zero_division=0来避免警告
    print(classification_report(targets, preds, 
                               target_names=[id2label[i] for i in range(len(id2label))],
                               zero_division=0))
    
    # 显示每个类别的预测分布
    unique, counts = np.unique(preds, return_counts=True)
    print("\n预测分布:")
    for i, count in zip(unique, counts):
        print(f"{id2label[i]}: {count}")
    
    # 显示真实标签分布
    true_unique, true_counts = np.unique(targets, return_counts=True)
    print("\n真实标签分布:")
    for i, count in zip(true_unique, true_counts):
        print(f"{id2label[i]}: {count}")


# ========= 主训练循环 =========
for epoch in range(5):  # 保持5轮训练
    print(f"\n[Epoch {epoch + 1}/5]")
    train_loss = train_one_epoch(model, train_loader)
    evaluate(model, test_loader)
    
    # 更新学习率调度器
    scheduler.step()
    
    if accelerator.is_main_process:
        torch.save(model.state_dict(), f"blip2_classifier_epoch{epoch + 1}.pth")

print("✅ 训练完成")