import os

def remove_substring_from_filenames(directory, substring):
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(directory):
        # 检查文件名中是否包含指定的子字符串
        if substring in filename:
            # 构造旧的文件路径
            old_file_path = os.path.join(directory, filename)
            # 构造新的文件名，去掉指定的子字符串
            new_filename = filename.replace(substring, '')
            # 构造新的文件路径
            new_file_path = os.path.join(directory, new_filename)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {old_file_path} -> {new_file_path}')

# 指定文件夹路径
directory_path = 'C:\\files\DeepLearning\dataset\images'
# 指定要去掉的子字符串
substring_to_remove = '微信图片_2024'

# 调用函数
remove_substring_from_filenames(directory_path, substring_to_remove)
