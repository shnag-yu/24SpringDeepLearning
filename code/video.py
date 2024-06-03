import cv2
import torch
import numpy as np

def load_models():
    model_waste = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5-master\\runs\\train\\exp4\\weights\\best.pt')
    model_gesture = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5-master\\runs\\train\\exp5\\weights\\best.pt')
    return model_waste, model_gesture

def handle_pre_phase(results_gesture):
    output_frame = np.squeeze(results_gesture.render())
    cv2.putText(output_frame, 'Welcome to Intelligent Waste Classification System!\nMake a OK-GESTURE to start.', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return output_frame

def handle_detecting_phase(results_waste):
    output_frame = np.squeeze(results_waste.render())
    cv2.putText(output_frame, 'Detecting waste...\nMake VICTORY-GESTURE to confirm the result\nFIST-GESTURE to manually sort', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return output_frame

def handle_sorting_phase(results_waste):
    output_frame = np.squeeze(results_waste.render())
    cv2.putText(output_frame, 'Sorting waste...\nMake a HAND-OPEN-GESTURE to quit', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return output_frame

def handle_manual_phase(results_waste):
    output_frame = np.squeeze(results_waste.render())
    cv2.putText(output_frame, 'Manual sorting...\nMake a HAND-OPEN-GESTURE to quit', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return output_frame

def handle_end_phase(results_gesture):
    output_frame = np.squeeze(results_gesture.render())
    cv2.putText(output_frame, 'Thank you!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return output_frame

def process_frame(frame, model_waste, model_gesture, phase):
    results_waste = model_waste(frame)
    results_gesture = model_gesture(frame)
    predictions_gesture = results_gesture.pred[0]

    for pred in predictions_gesture:
        pred_class = pred[-1]
        if pred_class == 0:
            phase = 'detecting'
        elif pred_class == 1:
            phase = 'sorting'
        elif pred_class == 2:
            phase = 'manual'
        elif pred_class == 3:
            phase = 'end'
    
    if phase == 'pre':
        output_frame = handle_pre_phase(results_gesture)
    elif phase == 'detecting':
        output_frame = handle_detecting_phase(results_waste)
    elif phase == 'sorting':
        output_frame = handle_sorting_phase(results_waste)
    elif phase == 'manual':
        output_frame = handle_manual_phase(results_waste)
    elif phase == 'end':
        output_frame = handle_end_phase(results_gesture)

    return output_frame, phase

def main():
    model_waste, model_gesture = load_models()

    video_path = 'videos\\0531182049.mp4'
    output_video_path = 'videos\\0531182049_pred.mp4'

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    phase = 'pre'
    it = 0

    while cap.isOpened():
        ret, frame = cap.read()
        it += 1
        if it % 10 == 0:
            print(f'processing frame {it}...')
        if not ret:
            break

        output_frame, phase = process_frame(frame, model_waste, model_gesture, phase)
        out.write(output_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
