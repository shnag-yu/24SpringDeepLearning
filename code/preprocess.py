import os
import random
import argparse
from pathlib import Path
 
def split_dataset(data_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # 获取所有图像文件
    image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
 
    # 随机打乱
    random.shuffle(image_files)
    total_count = len(image_files)
 
    # 划分数据集
    train_count = int(total_count * train_ratio)
    val_count = int(total_count * val_ratio)
 
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
 
    # 保存文件路径
    with open(output_dir / "train.txt", "w") as f:
        f.writelines(f"{str(file)}\n" for file in train_files)
 
    with open(output_dir / "val.txt", "w") as f:
        f.writelines(f"{str(file)}\n" for file in val_files)
 
    with open(output_dir / "test.txt", "w") as f:
        f.writelines(f"{str(file)}\n" for file in test_files)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, val and test.")
    parser.add_argument("data_dir", help="Directory containing images and labels.")
    parser.add_argument("output_dir", help="Directory to save train.txt, val.txt and test.txt.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio.")
    args = parser.parse_args()
 
    split_dataset(args.data_dir, args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio)