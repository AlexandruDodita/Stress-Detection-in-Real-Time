import cv2
import numpy as np
from deepface import DeepFace
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta


class StressDetector:
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Single person mode variables
        self.single_person_blinks = deque(maxlen=60)
        self.single_person_last_eye_state = False

        # Multiple person tracking
        self.face_data = defaultdict(lambda: {
            'blink_times': deque(maxlen=60),
            'last_eye_state': False,
            'stress_score': 0,
            'emotion_result': None,
            'last_emotion_time': 0
        })

        self.single_person_mode = True
        self.mode_switch_time = time.time()
        self.current_frame = None
        self.emotion_lock = threading.Lock()

    def get_face_id(self, face_location, frame_width):
        x, y, w, h = face_location
        return f"{x // 50}_{y // 50}"

    def detect_blinks(self, frame, face_location, face_id=None):
        x, y, w, h = face_location
        face_roi = frame[y:y + h, x:x + w]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray_roi, 1.1, 4)
        eyes_visible = len(eyes) > 0

        if self.single_person_mode:
            if not eyes_visible and self.single_person_last_eye_state:
                self.single_person_blinks.append(datetime.now())
            self.single_person_last_eye_state = eyes_visible
        else:
            if not eyes_visible and self.face_data[face_id]['last_eye_state']:
                self.face_data[face_id]['blink_times'].append(datetime.now())
            self.face_data[face_id]['last_eye_state'] = eyes_visible

        return len(eyes)

    def calculate_blink_rate(self, face_id=None):
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        if self.single_person_mode:
            return sum(1 for t in self.single_person_blinks if t > one_minute_ago)
        return sum(1 for t in self.face_data[face_id]['blink_times'] if t > one_minute_ago)

    def calculate_stress_score(self, emotions_dict, face_id=None):
        score = 0
        sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
        top_3_emotions = sorted_emotions[:3]
        emotion_names = [e[0] for e in top_3_emotions]

        if 'angry' in emotion_names[:2] and 'fear' in emotion_names[:2]:
            if emotion_names[2] in ['neutral', 'disgust']:
                score += 30
            else:
                score += 10

        score += emotions_dict.get('angry', 0) * 0.4
        score += emotions_dict.get('fear', 0) * 0.4
        if emotions_dict.get('happy', 100) < 10:
            score += (10 - emotions_dict['happy']) * 0.1

        blink_rate = self.calculate_blink_rate(face_id)
        blink_contribution = min((blink_rate - 20) * 2, 30)
        score += blink_contribution

        return max(0, min(100, score))

    def get_stress_level(self, score):
        if score < 25:
            return "No Stress", (0, 255, 0)
        elif score < 50:
            return "Mild Stress", (0, 255, 255)
        elif score < 75:
            return "Moderate Stress", (0, 165, 255)
        else:
            return "High Stress", (0, 0, 255)

    def analyze_emotions_thread(self):
        while True:
            with self.emotion_lock:
                if self.current_frame is None:
                    continue
                frame = self.current_frame.copy()

            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if not isinstance(analysis, list):
                    analysis = [analysis]

                with self.emotion_lock:
                    for face_data in analysis:
                        face_loc = face_data['region']
                        face_id = self.get_face_id((face_loc['x'], face_loc['y'],
                                                    face_loc['w'], face_loc['h']),
                                                   frame.shape[1])
                        self.face_data[face_id]['emotion_result'] = face_data
                        self.face_data[face_id]['last_emotion_time'] = time.time()
            except Exception:
                pass
            time.sleep(0.1)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        emotion_thread = threading.Thread(target=self.analyze_emotions_thread, daemon=True)
        emotion_thread.start()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with self.emotion_lock:
                self.current_frame = frame.copy()

            if time.time() - self.mode_switch_time < 2:
                mode_text = "Single-Person Mode" if self.single_person_mode else "Multiple-Person Mode"
                cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for face_location in faces:
                x, y, w, h = face_location
                face_id = self.get_face_id(face_location, frame.shape[1])

                self.detect_blinks(frame, face_location, face_id)

                if (self.face_data[face_id]['emotion_result'] and
                        time.time() - self.face_data[face_id]['last_emotion_time'] < 1.0):

                    emotions_dict = self.face_data[face_id]['emotion_result']['emotion']
                    stress_score = self.calculate_stress_score(emotions_dict,
                                                               face_id if not self.single_person_mode else None)

                    stress_level, color = self.get_stress_level(stress_score)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    y_offset = y - 10
                    cv2.putText(frame, f"{stress_level} ({stress_score:.1f})",
                                (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    y_offset -= 25
                    cv2.putText(frame,
                                f"Blinks: {self.calculate_blink_rate(face_id if not self.single_person_mode else None)}/min",
                                (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    for i, (emotion, score) in enumerate(
                            sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)[:3]):
                        y_offset -= 25
                        cv2.putText(frame, f"{emotion}: {score:.1f}%",
                                    (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow('Stress Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.single_person_mode = not self.single_person_mode
                self.mode_switch_time = time.time()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = StressDetector()
    detector.run()
