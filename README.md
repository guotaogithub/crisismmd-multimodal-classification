# Crisis Social Media Multimodal Classification with BLIP2

## 1. Project Overview

This project builds a multimodal classifier for social media posts related to disaster events. A frozen BLIP2 model is used to extract text and image features, followed by a lightweight MLP classifier for four-way classification.

### Target Classes

* `urgent_help`: Immediate rescue or emergency requests
* `relief_info`: Helpful information (shelters, donations, aid)
* `misinformation`: False or misleading content
* `irrelevant`: Unrelated or noisy posts

## 2. Dataset Construction

### Data Sources and Motivation

To construct a high-quality, balanced multimodal dataset suitable for fine-grained crisis post classification, we combined and filtered data from multiple open-source datasets:

| Dataset                 | Source URL                                                       | Purpose                                                                                           |
| ----------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **CrisisMMD v2.0**      | [https://crisisnlp.qcri.org/](https://crisisnlp.qcri.org/)       | Core source for tweet-image pairs with humanitarian intent categories                             |
| **MediaEval 2015/2016** | [http://www.multimediaeval.org/](http://www.multimediaeval.org/) | Image-verification corpus: used to collect mismatched or reused images for `misinformation` class |

We did **not** directly use the raw class labels provided by CrisisMMD or MediaEval. Instead, we **reclassified samples manually** into four custom categories aligned with real-world emergency response needs.

### Label Mapping Strategy

The new labels and their composition rules are:

* `urgent_help`: posts indicating people trapped, injured, or requesting immediate assistance
* `relief_info`: reports of aid delivery, resource availability, shelters, etc.
* `misinformation`: reused images from other events, visually mismatched or fake content
* `irrelevant`: spam, jokes, commentary, or non-crisis posts

### Dataset Construction Pipeline

1. **Data Extraction**:

   * Filtered CrisisMMD images with available English tweets and clear image links
   * Selected MediaEval samples with misleading image-text pairs

2. **Manual Re-labeling**:

   * Unified all data under the 4 new classes
   * Annotated 400 samples for development (100 per class)
   * Balanced remaining \~15,000 samples using stratified sampling

3. **Cleaning and Formatting**:

   * Verified image paths and resized corrupted ones to white-filled dummy images
   * Normalized text encoding (UTF-8), removed HTML/emoji noise
   * Saved as `.csv` files with consistent fields: `text`, `image_path`, `label`

### Directory Layout

```
dataset/
├── images_test/                      # Images for test set
├── images_train/                     # Images for training set
├── test_balanced.csv                 # Full-size balanced test set (~2.3k samples)
├── train_balanced.csv                # Full-size balanced training set (~13k samples)
├── test_balanced_100.csv             # Mini test set for fast dev (100 per class)
├── train_balanced_100.csv            # Mini training set (100 per class)
└── train_balanced_100_interleaved.csv # Interleaved variant (class-distributed order)
```

### CSV Format

| Column       | Type   | Description                                                          |
| ------------ | ------ | -------------------------------------------------------------------- |
| `text`       | string | Cleaned post content                                                 |
| `image_path` | string | Relative path to the associated image                                |
| `label`      | string | One of: `urgent_help`, `relief_info`, `misinformation`, `irrelevant` |

---

## 3. Model Architecture

A frozen `Salesforce/blip2-opt-2.7b-coco` model is used for visual-language embedding, followed by a lightweight classifier.

### Pipeline

```
BLIP2 (frozen)
↓
Extract embeddings (vision + text)
↓
Concatenate embeddings
↓
3-layer MLP classifier
↓
4-class output
```

### Highlights

* BLIP2 parameters are frozen
* Features fused via concatenation
* `CrossEntropyLoss` with auto or manual class weights
* Compatible with CUDA, Apple MPS, CPU
* Trained using HuggingFace `accelerate`

---

## 4. Evaluation Results

Model accuracy on `test_balanced_100.csv`: **38.75%**

### Confusion Matrix

|                        | Pred: Irrelevant | Pred: Misinformation | Pred: Relief Info | Pred: Urgent Help | Total Errors |
| ---------------------- | ---------------- | -------------------- | ----------------- | ----------------- | ------------ |
| Actual: Irrelevant     | **80**           | Error                | Error             | Error             | **20**       |
| Actual: Misinformation | Error            | **20**               | Error             | Error             | **80**       |
| Actual: Relief Info    | Error            | Error                | **35**            | Error             | **65**       |
| Actual: Urgent Help    | Error            | Error                | Error             | **20**            | **80**       |
| Misclassified as →     | **185**          | **31**               | **22**            | **7**             |              |

---

### Classification Report

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| irrelevant       | 0.30      | 0.80   | 0.44     | 100     |
| misinformation   | 0.39      | 0.20   | 0.26     | 100     |
| relief\_info     | 0.61      | 0.35   | 0.45     | 100     |
| urgent\_help     | 0.74      | 0.20   | 0.31     | 100     |
| **Macro Avg**    | **0.51**  | 0.39   | 0.37     | 400     |
| **Weighted Avg** | 0.51      | 0.39   | 0.37     | 400     |
| **Accuracy**     |           |        | **0.39** | 400     |

### Interpretation

* **Precision**: High for `urgent_help` and `relief_info`, suggesting good trust in those predictions.
* **Recall**: High for `irrelevant` (80%), but low (20%) for `urgent_help` and `misinformation`.
* **F1-Score**: All scores below 0.5, indicating challenges in achieving balance between precision and recall.

---


# 基于 BLIP2 的灾害社交媒体图文分类系统

## 一、项目简介

本项目构建了一个多模态分类系统，旨在自动识别灾害相关的社交媒体帖子类型。系统基于冻结的 BLIP2 模型提取文本与图像特征，并通过轻量级 MLP 分类器实现四分类。

### 目标分类：

* `urgent_help`：紧急求助信息（如救援、人员失踪）
* `relief_info`：提供援助与资源（如捐助、避难所）
* `misinformation`：错误或误导性信息
* `irrelevant`：与灾害无关的内容

---


## 二、数据集构建（Dataset Construction）

### 数据来源与构建动机

为了构建一个适合图文联合建模、符合灾害应急分类实际需求的高质量平衡数据集，我们整合并筛选了以下两个开源数据集：

| 数据集名称                   | 来源地址                                                             | 用途说明                                 |
| ----------------------- | ---------------------------------------------------------------- | ------------------------------------ |
| **CrisisMMD v2.0**      | [https://crisisnlp.qcri.org/](https://crisisnlp.qcri.org/)       | 提供灾害类推文及图像，是主要数据来源                   |
| **MediaEval 2015/2016** | [http://www.multimediaeval.org/](http://www.multimediaeval.org/) | 提供用于误导的信息验证图像，构造 `misinformation` 类别 |

> 注意：我们**并未直接使用原始标签**，而是基于图文内容重新定义并手动标注为4个统一类别，以适配模型训练目标。

---

### 标签重定义策略（Label Mapping）

重新定义的四个分类标签及其划分规则如下：

* `urgent_help`：含有紧急求助内容，如被困、伤员、失踪等；
* `relief_info`：救援资源信息，如物资发放、避难所开放、官方通报等；
* `misinformation`：错误图文匹配、旧图复用、虚假内容；
* `irrelevant`：与灾害无关的信息、评论、闲聊或噪声。

---

### 构建流程（Construction Steps）

1. **初步筛选：**

   * 从 CrisisMMD 中提取英文推文 + 图像样本；
   * 从 MediaEval 提取图文不一致的误导性样本；

2. **手动重标注：**

   * 所有样本根据内容重新标注为上述四类；
   * 构造小规模平衡数据（每类100条，共400条）用于快速开发；
   * 对剩余约1.5万样本进行分层采样，确保类别平衡性。

3. **数据清洗与标准化：**

   * 修复无效图片路径，损坏图用白图占位；
   * 文本统一编码、去除 HTML 与 emoji 噪声；
   * 最终保存为结构统一的 `.csv` 格式。

---

### 数据目录结构（Directory Structure）

```
dataset/
├── images_test/                      # 测试集图像
├── images_train/                     # 训练集图像
├── test_balanced.csv                 # 大型平衡测试集 (~2,332条)
├── train_balanced.csv                # 大型平衡训练集 (~13,177条)
├── test_balanced_100.csv             # 小型平衡测试集 (每类100条)
├── train_balanced_100.csv            # 小型训练集 (每类100条)
└── train_balanced_100_interleaved.csv # 小型交错训练集（类别轮换）
```

---

### CSV 文件结构（CSV Format）

| 字段名          | 类型  | 含义说明                                                             |
| ------------ | --- | ---------------------------------------------------------------- |
| `text`       | 字符串 | 清洗后的原始推文文本                                                       |
| `image_path` | 字符串 | 图像相对路径（相对于 dataset/）                                             |
| `label`      | 字符串 | 四个标签之一：`urgent_help`、`relief_info`、`misinformation`、`irrelevant` |


## 三、模型架构

本项目采用冻结的 `Salesforce/blip2-opt-2.7b-coco` 模型获取多模态特征，并使用自定义三层 MLP 分类器进行四分类任务。

### 流程结构：

```
BLIP2 (冻结)
↓
提取图像 + 文本嵌入
↓
拼接特征
↓
多层感知机分类器（MLP）
↓
四分类输出
```

### 特性：

* 模型参数全部冻结，减少过拟合风险
* 使用类权重平衡交叉熵损失函数
* 兼容多种平台（CUDA / MPS / CPU）
* 使用 HuggingFace `accelerate` 进行训练与优化

---

## 四、性能评估

在 `test_balanced_100.csv` 上最高准确率为 **38.75%**。

### 混淆矩阵（Confusion Matrix）

|                    | 预测为 Irrelevant | 预测为 Misinformation | 预测为 Relief Info | 预测为 Urgent Help | 该类别总误差 |
| ------------------ | -------------- | ------------------ | --------------- | --------------- | ------ |
| 实际为 Irrelevant     | **80**         | 误判                 | 误判              | 误判              | **20** |
| 实际为 Misinformation | 误判             | **20**             | 误判              | 误判              | **80** |
| 实际为 Relief Info    | 误判             | 误判                 | **35**          | 误判              | **65** |
| 实际为 Urgent Help    | 误判             | 误判                 | 误判              | **20**          | **80** |
| 被误判为该类总数           | **185**        | **31**             | **22**          | **7**           |        |

---

### 分类报告（Classification Report）

| 类别             | 精确率  | 召回率  | F1 分数    | 样本数 |
| -------------- | ---- | ---- | -------- | --- |
| irrelevant     | 0.30 | 0.80 | 0.44     | 100 |
| misinformation | 0.39 | 0.20 | 0.26     | 100 |
| relief\_info   | 0.61 | 0.35 | 0.45     | 100 |
| urgent\_help   | 0.74 | 0.20 | 0.31     | 100 |
| 宏平均            | 0.51 | 0.39 | 0.37     | 400 |
| 加权平均           | 0.51 | 0.39 | 0.37     | 400 |
| 总体准确率          |      |      | **0.39** | 400 |

### 指标解读

* **精确率**：`urgent_help` 与 `relief_info` 精确率较高，说明这些预测值得信赖。
* **召回率**：`irrelevant` 类别召回率高达 80%，但 `misinformation` 和 `urgent_help` 的召回率偏低，漏判问题突出。
* **F1 分数**：所有类别 F1 分数均低于 0.5，提示模型仍需改进，以实现精确性与全面性的平衡。

