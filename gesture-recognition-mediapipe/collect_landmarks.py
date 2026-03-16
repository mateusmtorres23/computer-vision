import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import argparse

MODEL_PATH = "models/gesture_recognizer.task"

def main():
    parser = argparse.ArgumentParser(description='Coleta de hand landmarks para dataset.')
    parser.add_argument('--label', type=str, required=True, help='Label do gesto que será coletado')
    parser.add_argument('--output', type=str, default='Data/hand_landmarks_data.csv', help='Arquivo CSV de saída')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"Erro: O arquivo {MODEL_PATH} não foi encontrado na pasta atual.")
        return

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2, 
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    file_exists = os.path.isfile(args.output)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.tasks.vision.HandLandmarksConnections
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles

    print(f"Coletando dados para a label: {args.label}")
    print("Comandos:\n 's' - Salva UMA captura\n 'r' - Inicia/Para gravação CONTÍNUA\n 'q' - Sair")

    recording = False

    with GestureRecognizer.create_from_options(options) as recognizer:
        with open(args.output, mode='a', newline='') as f:
            csv_writer = csv.writer(f)
            
            if not file_exists:
                header = ['label']
                for hand_type in ['Left', 'Right']:
                    for i in range(21):
                        header.extend([f'{hand_type}_x{i}', f'{hand_type}_y{i}', f'{hand_type}_z{i}'])
                csv_writer.writerow(header)

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                
                recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

                left_hand_lms = None
                right_hand_lms = None
                detected_hands_text = "Nenhuma"

                if recognition_result.hand_landmarks:
                    hands_detected = []
                    
                    for idx, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                        handedness = recognition_result.handedness[idx][0].category_name
                        hands_detected.append(handedness)
                        
                        if handedness == "Left":
                            left_hand_lms = hand_landmarks
                        elif handedness == "Right":
                            right_hand_lms = hand_landmarks
                            
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                    
                    detected_hands_text = " + ".join(hands_detected)

                def get_coords(landmarks):
                    if landmarks:
                        return [val for lm in landmarks for val in (lm.x, lm.y, lm.z)]
                    return [0.0] * (21 * 3)

                row_data = [args.label] + get_coords(left_hand_lms) + get_coords(right_hand_lms)

                if recording and (left_hand_lms or right_hand_lms):
                    csv_writer.writerow(row_data)

                status_color = (0, 255, 0) if recording else (255, 255, 255)
                status_text = "GRAVANDO" if recording else "STANDBY"
                cv2.putText(frame, f"Label: {args.label} [{status_text}]", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                ui_color = (0, 255, 0) if (left_hand_lms and right_hand_lms) else (0, 165, 255) if (left_hand_lms or right_hand_lms) else (0, 0, 255)
                cv2.putText(frame, f"Maos: {detected_hands_text}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_color, 2)

                cv2.imshow('Coleta de Dados - Landmark Recorder', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if left_hand_lms or right_hand_lms:
                        csv_writer.writerow(row_data)
                        f.flush()
                        print(f"Frame salvo para label '{args.label}'")
                        cv2.rectangle(frame, (0,0), (640,480), (0,255,0), 10)
                        cv2.imshow('Coleta de Dados - Landmark Recorder', frame)
                        cv2.waitKey(50)
                elif key == ord('r'):
                    recording = not recording
                    print(f"Gravação contínua: {'INICIADA' if recording else 'PARADA'}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()