# import os

# def process_files(directory):
#     # 遍历指定文件夹中的所有文件
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
        
#         # 确保是文件而不是文件夹
#         if os.path.isfile(file_path):
#             with open(file_path, 'r') as file:
#                 lines = file.readlines()
            
#             # 处理每一行的第一个数字
#             new_lines = []
#             for line in lines:
#                 parts = line.split()
#                 if parts:  # 确保这一行不是空行
#                     try:
#                         # 将第一个数字减去 15
#                         print(parts[0])
#                         first_number = int(int(parts[0]) - 15)
#                         parts[0] = str(first_number)
#                     except ValueError:
#                         # 如果第一个部分不是数字，跳过这一行
#                         pass
#                 new_lines.append(' '.join(parts))
            
#             # 将修改后的内容写回文件
#             with open(file_path, 'w') as file:
#                 file.write('\n'.join(new_lines) + '\n')

# # 指定文件夹路径
# directory_path = 'dataset\\labels\\val'

# # 调用函数
# process_files(directory_path)

import os
import xml.etree.ElementTree as ET

# 定义类别标签，根据需要修改
class_names = ['start', 'confirm', 'cancel', 'quit']

# 文件夹路径
input_folder = 'gesture_data/images/img'
output_folder = 'gesture_data/images/img'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有xml文件
for filename in os.listdir(input_folder):
    if filename.endswith('.xml'):
        # 解析XML文件
        tree = ET.parse(os.path.join(input_folder, filename))
        root = tree.getroot()

        # 获取图像的宽度和高度
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # 创建对应的txt文件
        txt_filename = filename.replace('.xml', '.txt')
        with open(os.path.join(output_folder, txt_filename), 'w') as f:
            # 遍历每个对象
            for obj in root.iter('object'):
                class_name = obj.find('name').text
                if class_name in class_names:
                    class_id = class_names.index(class_name)

                    # 获取边界框坐标
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)

                    # 转换为YOLO格式
                    x_center = (xmin + xmax) / 2.0 / width
                    y_center = (ymin + ymax) / 2.0 / height
                    bbox_width = (xmax - xmin) / width
                    bbox_height = (ymax - ymin) / height

                    # 写入文件
                    f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

print("转换完成")
