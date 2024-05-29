import os
configs = [
    'path: C:\\files\DeepLearning\dataset\images',  # 图片存放的主目录
    'train: train',  # 训练数据集的子目录
    'val: val',  # 交叉验证集的子目录
    'nc: 4',   # 检测任务需要检测的目标种类数
    'names: ["recyclable waste", "wet waste", "hazardous waste", "dry waste"]' 
]

# 将配置文件写入 yaml 文件，下面会用到
with open(os.path.join('C:\\files\DeepLearning\dataset', 'dataset.yaml'), 'w') as f:
    for config in configs:
        f.write(f'{config}\n')