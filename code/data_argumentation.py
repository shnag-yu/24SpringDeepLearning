import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os

def load_images_and_labels(image_dir, label_dir):
    images = []
    labels = []
    for image_file in os.listdir(image_dir):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, image_file.replace(".jpg", ".txt"))
            image = cv2.imread(image_path)
            with open(label_path, 'r') as f:
                label = f.read().strip()
            images.append((image, label))
    return images

def save_image_and_label(image, label, save_dir, file_name):
    image_save_path = os.path.join(save_dir, 'images', file_name + ".jpg")
    label_save_path = os.path.join(save_dir, 'labels', file_name + ".txt")
    cv2.imwrite(image_save_path, image)
    with open(label_save_path, 'w') as f:
        f.write(label)

def augment_and_save(images, save_dir, num_augmentations=5):
    transform = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=0, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MotionBlur(p=0.5),
        A.GaussNoise(p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', min_area=0.1, min_visibility=0.1, label_fields=[]))

    for i, (image, label) in enumerate(images):
        bboxes = [[float(x) for x in label.split()]]  # 将标签格式转换为浮点数列表
        for j in range(num_augmentations):
            transformed = transform(image=image, bboxes=bboxes)
            augmented_image = transformed['image']
            augmented_bboxes = transformed['bboxes']
            augmented_label = ' '.join(map(str, augmented_bboxes[0]))
            file_name = f"{i}_{j}"
            save_image_and_label(augmented_image, augmented_label, save_dir, file_name)

# 使用示例
image_dir = 'gesture_data/images'
label_dir = 'gesture_data/labels'
save_dir = 'augmented_data'

images = load_images_and_labels(image_dir, label_dir)
augment_and_save(images, save_dir)
