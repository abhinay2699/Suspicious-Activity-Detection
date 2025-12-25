import cv2
import numpy as np
import time

def predict_on_video(video_path, output_path, model, sequence_length, classes, height, width):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip = max(int(total_frames / sequence_length), 1)

    for i in range(sequence_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (height, width))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) < sequence_length:
        return "Video too short", 0.0

    start_time = time.time()
    prediction = model.predict(np.expand_dims(frames, axis=0))[0]
    inference_time = time.time() - start_time

    predicted_label = np.argmax(prediction)
    predicted_class_name = classes[predicted_label]
    confidence = float(prediction[predicted_label])

    # Annotate output video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, f"Prediction: {predicted_class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(frame)

    cap.release()
    out.release()

    return predicted_class_name, confidence
