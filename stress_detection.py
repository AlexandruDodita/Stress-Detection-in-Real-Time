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
        self.single_person_data = {
            'blink_times': deque(maxlen=60),
            'last_eye_state': False,
            'micro_movements': deque(maxlen=1800),  # 30 fps * 60 seconds = 1800 frames for 1 minute
            'prev_face_gray': None,
            'emotion_result': None,
            'last_emotion_time': 0
        }

        # Multiple person tracking
        self.face_data = defaultdict(lambda: {
            'blink_times': deque(maxlen=60),
            'last_eye_state': False,
            'micro_movements': deque(maxlen=1800),  # 30 fps * 60 seconds = 1800 frames for 1 minute
            'prev_face_gray': None,
            'emotion_result': None,
            'last_emotion_time': 0,
            'last_seen': time.time()
        })

        self.single_person_mode = True
        self.mode_switch_time = time.time()
        self.current_frame = None
        self.emotion_lock = threading.Lock()
        self.movement_threshold = 2.0
        self.standard_face_size = (128, 128)

    def get_face_id(self, face_location, frame_width):
        x, y, w, h = face_location
        return f"{x // 50}_{y // 50}"

    def clean_old_faces(self, max_age=5):
        """Remove faces that haven't been seen for a while"""
        current_time = time.time()
        faces_to_remove = [face_id for face_id, data in self.face_data.items()
                           if current_time - data['last_seen'] > max_age]
        for face_id in faces_to_remove:
            del self.face_data[face_id]

    def get_current_data(self, face_id=None):
        """Get the appropriate data dictionary based on current mode"""
        if self.single_person_mode:
            return self.single_person_data
        return self.face_data[face_id]

    def detect_micro_movements(self, face_roi, face_id=None):
        """
        Detects and quantifies micro movements in facial regions using optical flow.

        Parameters:
        - face_roi: Region of interest containing the face (BGR image)
        - face_id: Optional identifier for tracking multiple faces

        Returns:
        - float: Average movement magnitude over the recent time window
        """
        if face_roi is None or face_roi.size == 0:
            return 0

        current_data = self.get_current_data(face_id)
        face_roi_resized = cv2.resize(face_roi, self.standard_face_size)
        gray_roi = cv2.cvtColor(face_roi_resized, cv2.COLOR_BGR2GRAY)

        if current_data['prev_face_gray'] is None:
            current_data['prev_face_gray'] = gray_roi
            return 0

        try:
            # Calculate optical flow between consecutive frames
            # Parameters:
            # - prev and next frame
            # - None (calculated flow)
            # - 0.5 (pyramid scale)
            # - 3 (pyramid levels)
            # - 15 (window size)
            # - 3 (iterations)
            # - 5 (poly_n)
            # - 1.2 (poly_sigma)
            # - 0 (flags)
            flow = cv2.calcOpticalFlowFarneback(
                current_data['prev_face_gray'],
                gray_roi,
                None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Calculate movement magnitude using Pythagorean theorem
            # flow[..., 0] is x movement, flow[..., 1] is y movement
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            avg_movement = np.mean(magnitude)

            current_data['micro_movements'].append(avg_movement)

        except Exception as e:
            print(f"Error in micro movement detection: {e}")

        # used for next comparison
        current_data['prev_face_gray'] = gray_roi

        # calculate average movement over recent history
        recent_movements = list(current_data['micro_movements'])
        if not recent_movements:
            return 0

        minute_avg_movement = sum(recent_movements) / len(recent_movements)
        return minute_avg_movement

    def detect_blinks(self, frame, face_location, face_id=None):
        current_data = self.get_current_data(face_id)

        x, y, w, h = face_location
        face_roi = frame[y:y + h, x:x + w]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray_roi, 1.1, 4)
        eyes_visible = len(eyes) > 0

        if not eyes_visible and current_data['last_eye_state']:
            current_data['blink_times'].append(datetime.now())
        current_data['last_eye_state'] = eyes_visible

        return len(eyes)

    def calculate_blink_rate(self, face_id=None):
        current_data = self.get_current_data(face_id)
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        return sum(1 for t in current_data['blink_times'] if t > one_minute_ago)

    def calculate_stress_score(self, emotions_dict, face_id=None):
        score = 0
        sorted_emotions = sorted(emotions_dict.items(), key=lambda x: x[1], reverse=True)
        top_3_emotions = sorted_emotions[:3]
        emotion_names = [e[0] for e in top_3_emotions]

        # Emotional contribution (35% of total score)
        if 'angry' in emotion_names[:2] and 'fear' in emotion_names[:2]:
            if emotion_names[2] in ['neutral', 'disgust']:
                score += 17.5
            else:
                score += 7

        score += emotions_dict.get('angry', 0) * 0.25
        score += emotions_dict.get('fear', 0) * 0.25
        if emotions_dict.get('happy', 100) < 10:
            score += (10 - emotions_dict['happy']) * 0.05

        # Blink rate contribution (25% of total score)
        blink_rate = self.calculate_blink_rate(face_id)
        blink_contribution = min((blink_rate - 20) * 2, 25)
        score += blink_contribution

        # Movement contribution (40% of total score)
        avg_movement = self.detect_micro_movements(None, face_id)  # Pass None as we don't need new detection
        movement_score = min((avg_movement - self.movement_threshold) * 10, 40)
        score += max(0, movement_score)

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
                    if self.single_person_mode and analysis:
                        self.single_person_data['emotion_result'] = analysis[0]
                        self.single_person_data['last_emotion_time'] = time.time()
                    else:
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

            # Clean up old face data in multiple person mode
            if not self.single_person_mode:
                self.clean_old_faces()

            for face_location in faces:
                x, y, w, h = face_location
                face_id = None if self.single_person_mode else self.get_face_id(face_location, frame.shape[1])

                if not self.single_person_mode:
                    self.face_data[face_id]['last_seen'] = time.time()
                face_roi = frame[y:y + h, x:x + w]

                # Detect micro-movements and blinks
                micro_movement = self.detect_micro_movements(face_roi, face_id)
                self.detect_blinks(frame, face_location, face_id)

                current_data = self.get_current_data(face_id)

                if (current_data['emotion_result'] and
                        time.time() - current_data['last_emotion_time'] < 1.0):

                    emotions_dict = current_data['emotion_result']['emotion']
                    stress_score = self.calculate_stress_score(emotions_dict, face_id)

                    stress_level, color = self.get_stress_level(stress_score)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Display information
                    y_offset = y - 10
                    cv2.putText(frame, f"{stress_level} ({stress_score:.1f})",
                                (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    y_offset -= 25
                    cv2.putText(frame, f"Blinks: {self.calculate_blink_rate(face_id)}/min",
                                (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    y_offset -= 25
                    cv2.putText(frame, f"Movement: {micro_movement:.2f}",
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
                # Reset data when switching modes
                if self.single_person_mode:
                    self.single_person_data = {
                        'blink_times': deque(maxlen=60),
                        'last_eye_state': False,
                        'micro_movements': deque(maxlen=1800),
                        'prev_face_gray': None,
                        'emotion_result': None,
                        'last_emotion_time': 0
                    }

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = StressDetector()
    detector.run()
