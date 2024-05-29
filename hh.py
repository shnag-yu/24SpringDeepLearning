import os

def process_files(directory):
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # 确保是文件而不是文件夹
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # 处理每一行的第一个数字
            new_lines = []
            for line in lines:
                parts = line.split()
                if parts:  # 确保这一行不是空行
                    try:
                        # 将第一个数字减去 15
                        print(parts[0])
                        first_number = int(int(parts[0]) - 15)
                        parts[0] = str(first_number)
                    except ValueError:
                        # 如果第一个部分不是数字，跳过这一行
                        pass
                new_lines.append(' '.join(parts))
            
            # 将修改后的内容写回文件
            with open(file_path, 'w') as file:
                file.write('\n'.join(new_lines) + '\n')

# 指定文件夹路径
directory_path = 'dataset\\labels\\val'

# 调用函数
process_files(directory_path)
