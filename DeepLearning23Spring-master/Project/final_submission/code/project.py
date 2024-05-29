import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
import face_recognition
import datetime
import csv
import os
from PIL import Image, ImageDraw, ImageFont

from scipy.spatial import distance as dist
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import random
import math

import os, sys
os.chdir(sys.path[0])
weights_path = 'D:\FDU\Sophomore_semester2\Deep_learning\Project\\final_submission\input\\best.weights'
configuration_path = 'D:\FDU\Sophomore_semester2\Deep_learning\Project\\final_submission\input\yolov3.cfg'
probability_minimum = 0.5
threshold = 0.3

def get_model(weights_path, configuration_path):
    return cv2.dnn.readNetFromDarknet(configuration_path, weights_path)

# 实例化模型
network = get_model(weights_path, configuration_path)

# 获取所有层的 list
layers_names_all = network.getLayerNames()

# 加载 COCO labels
labels = open('D:\FDU\Sophomore_semester2\Deep_learning\Project\\final_submission\input\coco.names').read().strip().split('\n')

# 获取 ouput 层
layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
# Check point
print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

def detect_img(blob, frame, shape, left, top):
    network.setInput(blob)  # 将 blob 作为网络输入

    # 查看 forward 运行时间
    # start = time.time()
    output_from_network = network.forward(layers_names_output)
    # end = time.time()
    # print('YOLO v3 took {:.5f} seconds'.format(end - start))

    # 为每个 label 获取随机颜色以标识
    np.random.seed(42)
    # randint(low, high=None, size=None, dtype='l')
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # 为每个检测到的物体初始化 list
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # 获取原 input 图片的维度
    h, w = shape[:2]
    # h *= 480 // shape[0]
    # w *= 640 // shape[1]

    for result in output_from_network:
    # 遍历 output 层的所有检测对象
        for detection in result:
            # 获取当前对象的 class
            scores = detection[5:]
            class_current = np.argmax(scores)

            # 获取当前对象的置信度
            confidence_current = scores[class_current]

            # 只保留置信度大于阈值的对象
            if confidence_current > probability_minimum:
                # 将 bounding box 缩放以适应原来的 input 图片
                box_current = detection[0:4] * np.array([w, h, w, h])

                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2)) + left
                y_min = int(y_center - (box_height / 2)) + top
                
                print("x_min: ", x_min)
                print("y_min: ", y_min)

                # 将结果添加到准备好的 list 中
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    # 使用非极大值抑制来获得最终结果 bounding box
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    for i in range(len(class_numbers)):
        print(labels[int(class_numbers[i])])

    # 绘画 bounding box
    if len(results) > 0:
        # 遍历 results
        for i in results.flatten():
            # 获取当前 bounding box 坐标
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # 获取当前 bounding box 颜色
            colour_box_current = [int(j) for j in colours[class_numbers[i]]]

            # 绘图
            cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height),
                        colour_box_current, 5)

            # 输出 label 以及其置信度
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], confidences[i])

            # 将文字加入
            cv2.putText(frame, text_box_current, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, colour_box_current, 2)
            
    return frame

def save_custom_gesture(path, to_path):
    # 初始化MediaPipe的手部识别模型
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75)

    # 设置绘制工具
    mp_drawing = mp.solutions.drawing_utils

    # 指定包含图片的文件夹路径
    folder_path = path

    # 遍历文件夹中的所有图片文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 构造图片的完整路径
            image_path = os.path.join(folder_path, filename)

            # 加载图像
            image = cv2.imread(image_path)

            # 将图像从BGR转换为RGB格式（MediaPipe使用RGB格式）
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 执行手部关键点识别
            results = hands.process(image_rgb)

            # 绘制关键点
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 保存识别后的图片
            output_path = os.path.join(to_path, 'recognized_' + filename)
            cv2.imwrite(output_path, image)



path = "D:\FDU\Sophomore_semester2\Deep_learning\Project\\final_submission\code\custom_gesture"
to_path = "D:\FDU\Sophomore_semester2\Deep_learning\Project\\final_submission\code\\recognized_custom_gesture"
save_custom_gesture(path, to_path)

def load_recognized_hand_gesture(path):
    gesture_results = []  # 创建一个空列表，用于保存每个图片的识别结果

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75)
    
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(path, filename)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            # 将手势识别的结果保存到gesture_results列表中
            gesture_results.append(results.multi_hand_landmarks[0])

    return gesture_results

# 运行函数得到个性化手势信息
custom_gesture_info = load_recognized_hand_gesture(path) # 0-清除、1-加入购物车、2-购买

def calculate_angle(vector1, vector2):
    # 计算两个向量的点积
    dot_product = sum([a * b for a, b in zip(vector1, vector2)])
    # 计算两个向量的模长
    magnitude1 = math.sqrt(sum([a * a for a in vector1]))
    magnitude2 = math.sqrt(sum([a * a for a in vector2]))

    # 计算夹角（弧度）
    angle_radians = math.acos(dot_product / (magnitude1 * magnitude2))

    # 转换为角度
    angle_degrees = math.degrees(angle_radians)

    return angle_degrees

def get_finger_angles(hand_info):
    # 获取手指关键点的坐标
    thumb_tip = hand_info.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_info.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_info.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_info.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_info.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
    wrist = hand_info.landmark[mp.solutions.hands.HandLandmark.WRIST]

    # 计算手指之间的夹角
    thumb_angle = calculate_angle([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y, thumb_tip.z - wrist.z],
                                  [index_finger_tip.x - wrist.x, index_finger_tip.y - wrist.y, index_finger_tip.z - wrist.z])
    index_angle = calculate_angle([index_finger_tip.x - wrist.x, index_finger_tip.y - wrist.y, index_finger_tip.z - wrist.z],
                                  [middle_finger_tip.x - wrist.x, middle_finger_tip.y - wrist.y, middle_finger_tip.z - wrist.z])
    middle_angle = calculate_angle([middle_finger_tip.x - wrist.x, middle_finger_tip.y - wrist.y, middle_finger_tip.z - wrist.z],
                                   [ring_finger_tip.x - wrist.x, ring_finger_tip.y - wrist.y, ring_finger_tip.z - wrist.z])
    ring_angle = calculate_angle([ring_finger_tip.x - wrist.x, ring_finger_tip.y - wrist.y, ring_finger_tip.z - wrist.z],
                                 [pinky_tip.x - wrist.x, pinky_tip.y - wrist.y, pinky_tip.z - wrist.z])

    return thumb_angle, index_angle, middle_angle, ring_angle

def compare_hand_gesture(recognized_hand_info, present_hand_info):
    # 获取识别手势和当前手势的手指夹角
    recognized_angles = get_finger_angles(recognized_hand_info)
    present_angles = get_finger_angles(present_hand_info)

    # 设定夹角阈值
    angle_threshold = 15  # 可根据需要进行调整

    # 判断手势
    if all(abs(recognized_angles[i] - present_angles[i]) <= angle_threshold for i in range(4)):
        return True
    else:
        return False

def recognize_buy(hand_landmarks, isCustom):
    if isCustom:
        return compare_hand_gesture(hand_landmarks, custom_gesture_info[1])
    # 获取各个手指关键点的坐标
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]

    # 计算拇指和食指之间的距离
    distance = np.linalg.norm([thumb_tip.x - index_finger_tip.x, thumb_tip.y - index_finger_tip.y, thumb_tip.z - index_finger_tip.z])

    # 计算其他手指与腕部的距离
    thumb_to_palm = np.linalg.norm([thumb_tip.x - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x,
                                    thumb_tip.y - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y,
                                    thumb_tip.z - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].z])
    ring_to_palm = np.linalg.norm([ring_finger_tip.x - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x,
                                ring_finger_tip.y - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y,
                                ring_finger_tip.z - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].z])
    pinky_to_palm = np.linalg.norm([pinky_tip.x - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x,
                                pinky_tip.y - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y,
                                pinky_tip.z - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].z])
    distance_to_palm = thumb_to_palm + ring_to_palm + pinky_to_palm
    
    # print("dis", distance)
    # print("palm", distance_to_palm)
    # 设置"OK手势"的阈值
    threshold_distance = 0.08
    threshold_palm = 0.67

    if distance < threshold_distance and distance_to_palm > threshold_palm:
        # 检测到"OK手势"
        return True
    else:
        # 非"OK手势"
        return False

def recognize_pay(hand_landmarks, isCustom):
    if isCustom:
        return compare_hand_gesture(hand_landmarks, custom_gesture_info[2])
    # 获取各个手指关键点的坐标
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]

    # 计算手指之间的相对位置关系
    index_finger_dis = np.linalg.norm([index_finger_tip.x - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x,
                                         index_finger_tip.y - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y,
                                         index_finger_tip.z - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].z])
    middle_finger_dis = np.linalg.norm([middle_finger_tip.x - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x,
                                         middle_finger_tip.y - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y,
                                         middle_finger_tip.z - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].z])


    # 计算其他手指与腕部的距离
    thumb_to_palm = np.linalg.norm([thumb_tip.x - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x,
                                    thumb_tip.y - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y,
                                    thumb_tip.z - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].z])
    ring_to_palm = np.linalg.norm([ring_finger_tip.x - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x,
                                ring_finger_tip.y - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y,
                                ring_finger_tip.z - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].z])
    pinky_to_palm = np.linalg.norm([pinky_tip.x - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].x,
                                pinky_tip.y - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].y,
                                pinky_tip.z - hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST].z])
    distance_to_palm = thumb_to_palm + ring_to_palm + pinky_to_palm
    
    # 设置阈值
    threshold_palm = 0.65


    index_finger_second_joint = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
    standard = np.linalg.norm([index_finger_tip.x - index_finger_second_joint.x,
                                                        index_finger_tip.y - index_finger_second_joint.y,
                                                        index_finger_tip.z - index_finger_second_joint.z])

    is_index_up = False
    is_middle_up = False

    if index_finger_dis > 4 * standard and index_finger_dis > 0.25:
        # print("index ok!")
        is_index_up = True
    if middle_finger_dis > 4 * standard and middle_finger_dis > 0.25:
        # print("middle ok!")
        is_middle_up = True
    # print("index", is_index_finger_up)
    # print("middle", is_middle_finger_up)
    # print(distance_to_palm)

    # 判断手势是否符合 "耶" 手势的条件
    if is_index_up and is_middle_up and distance_to_palm < threshold_palm:
        # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return True
    else:
        return False

def faces_paths(folder):
    faces_list = []
    for path, subdirs, files in os.walk(folder):
        for name in files:
            if not name.startswith('.'):
                faces_list.append( os.path.join(path, name))
    return faces_list

def load_feature_face(picture_list):
    a_encodings =[]
    for a in picture_list:
        file = face_recognition.load_image_file(a)
        if file is None: 
            continue
        face_locations = face_recognition.face_locations(file)
        encodings = face_recognition.face_encodings(file, face_locations)

        if encodings is None or len(encodings)<=0: 
            continue
        a_encodings.append(encodings[0])
    return a_encodings

# Get the known face encodings of all the pic in directory Name
print("[INFO] encoding features of the known faces in directory /Name...")
textColor = (255, 0, 0)

mypath_a = "D:/FDU/Sophomore_semester2/Deep_learning/Project/final_submission/code/Name"
a_list = faces_paths(mypath_a)
known_face_encodings = load_feature_face(a_list)
known_face_names = a_list

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


scale_ratio = 0.5
font = ImageFont.truetype('D:/FDU/Sophomore_semester2/Deep_learning/Project/final_submission/code/SimSun.ttf', 15, encoding="utf-8")

checking_name = False
checking_name_count = 130

def face_recognition_module(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=scale_ratio, fy=scale_ratio)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    isMatch = False
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,0.4)
        name = "Name/未知.png"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)


            name = known_face_names[first_match_index]
            isMatch = True

        face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            sr = int(1/scale_ratio)
            top *= sr
            right *= sr
            bottom *= sr
            left *= sr

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            board_height = int((bottom-top)/5)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - board_height), (right, bottom), (0, 0, 255), cv2.FILLED)

            lable_name = name[-17:-4]
            frame_pil = Image.fromarray(frame)  #转为PIL的图片格式
            text_size = font.getsize(lable_name)
            ImageDraw.Draw(frame_pil).text((left + ((right-left)-text_size[0])/2, bottom - board_height+(board_height-text_size[1])/2), lable_name, (255, 255, 255), font)
            frame = np.array(frame_pil)
    if isMatch:
        return frame, isMatch, lable_name
    return frame, isMatch, None

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75)

# 创建空白画布
buffer = np.zeros((480,640, 3), np.uint8)

# 初始画笔位置
prev_x, prev_y = None, None

# 食指悬停开启绘图相关变量
hover_duration = 0
hover_duration_threshold = 100  # 设置悬停的时长
hover_distance_threshold = 6 # 设置悬停识别的敏感度
enable_drawing = False  # 是否开始绘图

index_move_distance = 100

# 方框绘画坐标
left_top_corner = None
right_bottom_corner = None

# 购买状态
state_buy = False

# 个性化状态
isCustom = False

cap = cv2.VideoCapture(0)

cropped_image = None

while True:
    ######################################## 手部关键点识别 ######################################### 
    ret,frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 因为摄像头是镜像的，所以将摄像头水平翻转
    # 不是镜像的可以不翻转
    frame= cv2.flip(frame,1)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 关键点可视化
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 计算手部框的坐标
            height, width, _ = frame.shape
            xmin, ymin, xmax, ymax = width, height, 0, 0

            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * width), int(landmark.y * height)
                xmin = min(xmin, x)
                ymin = min(ymin, y)
                xmax = max(xmax, x)
                ymax = max(ymax, y)

            # 绘制手部框
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # 获取食指坐标
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pinky_finger = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # 将食指坐标转换为画面坐标
            x = int(index_finger.x * frame.shape[1])
            y = int(index_finger.y * frame.shape[0])

            # 计算食指移动距离判断是否悬停并切换绘画状态
            if prev_x and prev_y: 
                index_move_distance = np.linalg.norm([x - prev_x, y-prev_y]);
                if index_move_distance < hover_distance_threshold and hover_duration < hover_duration_threshold:
                    hover_duration += 1
                else:
                    hover_duration = 0
            if hover_duration >= hover_duration_threshold:
                if enable_drawing:
                    right_bottom_corner = (x, y)
                    print("right_bottom_corner:", right_bottom_corner)
                    cv2.rectangle(buffer, left_top_corner, right_bottom_corner, (0, 255, 0), 5)

                    ################################# 商品目标检测 ##################################
                    left = min(left_top_corner[0], right_bottom_corner[0])
                    top = min(left_top_corner[1], right_bottom_corner[1])
                    right = max(left_top_corner[0], right_bottom_corner[0])
                    bottom = max(left_top_corner[1], right_bottom_corner[1])
            
                    cropped_image = frame[top:bottom, left:right]

                    # 获取帧图像的blob
                    print("Cropped Image Shape:", cropped_image.shape)
                    print("Cropped Image Size:", cropped_image.size)

                    if cropped_image.size != 0:
                        # cropped_image = cv2.resize(cropped_image, (128, 128))
                        blob = cv2.dnn.blobFromImage(cropped_image, 1 / 255.0, (128, 128), swapRB=True, crop=False)
                        # 应用目标检测
                        buffer = detect_img(blob, buffer, cropped_image.shape, left, top)
                    ################################ 商品目标检测结束 ###############################
                else:
                    left_top_corner = (x, y)
                    print("left_top_corner:", left_top_corner)
                enable_drawing = not enable_drawing
                hover_duration = 0

            #################################### 手势识别 1：清除画面 ###############################
            # 计算食指和小指之间的距离，识别握拳以刷新绘画痕迹
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            distance = np.linalg.norm([thumb_tip.x - pinky_finger.x, thumb_tip.y - pinky_finger.y, thumb_tip.z - pinky_finger.z])
            # 设置握拳的阈值
            threshold_clear = 0.15
            if distance < threshold_clear:
                # 手部握拳，清除绘画缓冲区
                buffer = np.zeros((480,640, 3), np.uint8)
            ################################### 手势识别 1 结束 #####################################
            ################################### 手势识别 2：购买 ####################################
            if recognize_buy(hand_landmarks, isCustom):
                buy_text = "Successful added to cart!"
                cv2.putText(buffer, buy_text, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            ################################## 手势识别 2 结束 ######################################
            ################################## 手势识别 3：付款 #####################################
            if recognize_pay(hand_landmarks, isCustom):
                pay_text = "Please pay the bill by face recognition!"
                buffer = np.zeros((480,640, 3), np.uint8)
                cv2.putText(buffer, pay_text, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                state_buy = True
            ################################## 手势识别 3 结束 ######################################
            #若状态为可绘画，则绘制
            if enable_drawing:
                # 如果之前有记录的上一个位置，则绘制线段
                if prev_x is not None and prev_y is not None:
                    cv2.line(buffer, (prev_x, prev_y), (x, y), (0, 0, 255), 5)

            # 更新上一个位置为当前位置
            prev_x, prev_y = x, y

            # 提示目前绘画开启状态
            if index_move_distance < hover_distance_threshold and hover_duration > 10:
                text = "Hovering" + str(hover_duration)
            elif enable_drawing:
                text = "Drawing Enabled"
            else:
                text = "Drawing Disabled"
            cv2.putText(frame, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    else:
        prev_x, prev_y = None, None

    # 将缓冲区中的线条叠加到摄像头画面上（若状态为可绘画）
    # frame = cv2.add(frame, buffer)
    # cv2.imshow('Project', frame)

    if state_buy:
        frame, isMatch, user = face_recognition_module(frame)
        print("in")
        # cv2.imshow('Project', frame)
        if isMatch:
            print("match")
            finish_text = "Payment complete! " + user
            cv2.putText(buffer, finish_text, (90, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            frame = cv2.add(frame, buffer)
            cv2.imshow('Project', frame)
            state_buy = False
    # else:
    #     cv2.imshow('Project', frame)

    frame = cv2.add(frame, buffer)
    cv2.imshow('Project', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()






