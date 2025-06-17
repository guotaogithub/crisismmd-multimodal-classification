import os
import pandas as pd

# 定义4个新标签及原始标签映射关系
STRICT_LABEL_MAPPING = {
    "emergency_response": [
        "affected_individuals",
        "injured_or_dead_people",
        "missing_or_found_people",
        "rescue_volunteering_or_donation_effort"
    ],
    "damage_report": [
        "infrastructure_and_utility_damage",
        "vehicle_damage"
    ],
    "information_share": [
        "other_relevant_information"
    ],
    "irrelevant": [
        "not_humanitarian"
    ]
}

DATA_DIR = "/Users/guotao/Documents/TextAndImage/FinalExam/CrisisMMD_v2.0"
IMAGE_DIR = os.path.join(DATA_DIR, "data_image")
TRAIN_TSV = os.path.join(DATA_DIR, "crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv")
DEV_TSV = os.path.join(DATA_DIR, "crisismmd_datasplit_all/task_humanitarian_text_img_dev.tsv")
TEST_TSV = os.path.join(DATA_DIR, "crisismmd_datasplit_all/task_humanitarian_text_img_test.tsv")

def verify_image_exists(image_path):
    """验证图片文件是否存在"""
    # 移除 'data_image/' 前缀
    if image_path.startswith('data_image/'):
        image_path = image_path[11:]  # 移除 'data_image/' 前缀
    full_path = os.path.join(IMAGE_DIR, image_path)
    return os.path.exists(full_path)

# 处理所有数据集
for tsv_file in [TRAIN_TSV, DEV_TSV, TEST_TSV]:
    # 读取数据
    df = pd.read_csv(tsv_file, sep='\t')
    
    # 打印数据集名称
    print(f"\n处理数据集: {os.path.basename(tsv_file)}")
    
    # 为每个新标签筛选数据
    sampled_dfs = []
    for new_label, old_labels in STRICT_LABEL_MAPPING.items():
        # 获取匹配的数据
        subset = df[df['label_text'].isin(old_labels)]
        if len(subset) > 0:
            # 验证图片存在性
            valid_subset = subset[subset['image'].apply(verify_image_exists)]
            print(f"{new_label} 类别有效数据数量: {len(valid_subset)}")
            
            if len(valid_subset) > 0:
                # 随机采样50条
                sampled_df = valid_subset.sample(n=min(50, len(valid_subset)), random_state=42)
                # 添加新列
                sampled_df['new_cate'] = new_label
                sampled_dfs.append(sampled_df)
                print(f"{new_label} 类别采样数量: {len(sampled_df)}")
            else:
                print(f"警告：{new_label} 类别没有有效的图片数据！")
    
    if sampled_dfs:
        # 合并所有采样的数据
        balanced_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42)
        
        # 生成新文件名
        base_name = os.path.basename(tsv_file)
        new_name = base_name.replace('.tsv', '_balanced.tsv')
        new_path = os.path.join(os.path.dirname(tsv_file), new_name)
        
        # 保存新数据集
        balanced_df.to_csv(new_path, sep='\t', index=False)
        print(f"\n处理完成: {new_name}")
        print("新标签分布:")
        print(balanced_df['new_cate'].value_counts())
    else:
        print("警告：没有找到任何符合条件的数据！") 