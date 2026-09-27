import cv2
import numpy as np
import torch
import time
from face_detection import RetinaFace as TorchRetinaFace

EMOTION_MODEL_PATH = r"D:\viyaltech\Project_shop_phase_17072026\Vcet_medical project\project\emotion_model_best.pth"
IMG_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CAM_WIDTH = 640
CAM_HEIGHT = 480
DETECT_EVERY_N_FRAMES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using: {DEVICE}")

def load_emotion_model(checkpoint_path, device):
    try:
        print(f"[INFO] Loading emotion model...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        class_names = checkpoint.get("class_names", None)
        if not class_names:
            return None, None
        
        from transformers import Swinv2ForImageClassification
        model = Swinv2ForImageClassification.from_pretrained(
            "microsoft/swinv2-tiny-patch4-window8-256",
            num_labels=len(class_names),
            ignore_mismatched_sizes=True
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.to(device)
        model.eval()
        
        print(f"[INFO] ✅ Model ready!")
        return model, class_names
    except Exception as e:
        print(f"[ERROR] {e}")
        return None, None

def preprocess_face_for_emotion(face_bgr, img_size=IMG_SIZE):
    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1).to(face_tensor.device)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1).to(face_tensor.device)
        face_tensor = (face_tensor - mean) / std
        return face_tensor.unsqueeze(0)
    except:
        return None

def predict_emotion(face_bgr, model, class_names, device):
    if model is None:
        return "neutral", 0.5
    try:
        face_tensor = preprocess_face_for_emotion(face_bgr)
        if face_tensor is None:
            return "neutral", 0.5
        
        face_tensor = face_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence, pred_idx = torch.max(probs, dim=-1)
            emotion = class_names[pred_idx.item()]
            conf_val = confidence.item()
            return emotion, conf_val
    except:
        return "neutral", 0.5

class AdvancedFacialAnalyzer:
    def __init__(self):
        self.eye_closed_frames = 0
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def analyze_eyes(self, frame, face_box):
        try:
            x1, y1, x2, y2 = face_box
            face_roi = frame[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return {'open': True, 'percent': 100, 'pupils': [], 'closed_time': 0, 'stress': False, 'brightness': 0}
            
            if len(face_roi.shape) == 3:
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_roi
            
            eyes = self.eye_cascade.detectMultiScale(gray_face, 1.05, 5, minSize=(20, 20), maxSize=(120, 120))
            
            eyes_open = False
            avg_brightness = 0
            pupils = []
            
            if len(eyes) >= 1:
                total_brightness = 0
                eye_count = 0
                
                for (ex, ey, ew, eh) in eyes[:2]:
                    eye_roi = gray_face[ey:ey+eh, ex:ex+ew]
                    brightness = cv2.mean(eye_roi)[0]
                    total_brightness += brightness
                    eye_count += 1
                    
                    _, thresh = cv2.threshold(eye_roi, 85, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    
                    if len(contours) > 0:
                        largest = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest)
                        if area > 20:
                            M = cv2.moments(largest)
                            if M['m00'] > 0:
                                cx = int(M['m10'] / M['m00']) + ex + x1
                                cy = int(M['m01'] / M['m00']) + ey + y1
                                pupils.append((cx, cy, int(area)))
                
                avg_brightness = total_brightness / eye_count if eye_count > 0 else 0
                
                OPEN_THRESHOLD = 100
                CLOSED_THRESHOLD = 70
                
                if avg_brightness > OPEN_THRESHOLD:
                    eyes_open = True
                    self.eye_closed_frames = 0
                elif avg_brightness < CLOSED_THRESHOLD:
                    eyes_open = False
                    self.eye_closed_frames += 1
                else:
                    eyes_open = self.eye_closed_frames == 0
                    if not eyes_open:
                        self.eye_closed_frames += 1
            else:
                eyes_open = False
                self.eye_closed_frames += 1
            
            closed_time = self.eye_closed_frames / 30.0
            stress_from_eyes = closed_time >= 1.5
            
            return {
                'open': eyes_open,
                'percent': 100 if eyes_open else 0,
                'pupils': pupils,
                'closed_time': closed_time,
                'stress': stress_from_eyes,
                'brightness': avg_brightness
            }
        except Exception as e:
            return {'open': True, 'percent': 100, 'pupils': [], 'closed_time': 0, 'stress': False, 'brightness': 0}
    
    def analyze_mouth(self, frame_bgr, face_box):
        try:
            x1, y1, x2, y2 = face_box
            face_roi = frame_bgr[y1:y2, x1:x2]
            
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 20, 0], dtype=np.uint8)
            upper = np.array([180, 255, 100], dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                
                if area > 50:
                    x, y, w, h = cv2.boundingRect(largest)
                    if w > 0:
                        ratio = h / float(w)
                        mouth_open = ratio > 0.4
                        smile = w > 30
                        return {'open': mouth_open, 'ratio': ratio, 'width': w, 'smile': smile}
            
            return {'open': False, 'ratio': 0, 'width': 0, 'smile': False}
        except:
            return {'open': False, 'ratio': 0, 'width': 0, 'smile': False}

class StressCalculator:
    def __init__(self):
        pass
    
    def calculate(self, emotion, confidence, eyes_analysis, mouth_analysis):
        emotion_scores = {
            'anger': 90, 'fear': 85, 'sadness': 75, 'sad': 75, 'contempt': 70,
            'disgust': 70, 'surprise': 55, 'neutral': 40, 'happiness': 10, 'happy': 10,
        }
        
        base = emotion_scores.get(emotion.lower(), 50)
        model_score = base * confidence
        
        if emotion and emotion.lower() not in ['neutral', 'happiness', 'happy']:
            print(f"[EMOTION] {emotion} ({confidence*100:.0f}%) → score: {model_score:.0f}")
        
        if eyes_analysis and eyes_analysis.get('stress', False):
            print(f"[STRESS] Eyes closed: {eyes_analysis['closed_time']:.1f}s")
            return 85, f"Eyes closed {eyes_analysis['closed_time']:.1f}s → STRESS", "EyeClosure"
        
        if mouth_analysis and mouth_analysis.get('smile', False):
            print(f"[NO-STRESS] Smile detected")
            return 15, "SMILE detected → NO STRESS", "Smile"
        
        if emotion and confidence > 0.4:
            emotion_lower = emotion.lower()
            if emotion_lower in ['anger', 'fear', 'sadness', 'sad', 'contempt', 'disgust', 'angry', 'afraid', 'disgusted']:
                print(f"[STRESS] {emotion} detected ({confidence*100:.0f}%)")
                if emotion_lower in ['anger', 'fear', 'angry', 'afraid']:
                    return 80, f"{emotion.upper()} → STRESS", "NegativeEmotion"
                else:
                    return 70, f"{emotion.upper()} → STRESS", "NegativeEmotion"
        
        if confidence > 0.7:
            final = model_score * 0.9 + 40 * 0.1
            weight_str = "Model(90%)"
        elif confidence > 0.4:
            final = model_score * 0.6 + 40 * 0.4
            weight_str = "Balanced(60-40)"
        else:
            final = model_score * 0.3 + 40 * 0.7
            weight_str = "Facial(70%)"
        
        return min(max(final, 0), 100), "", weight_str


def draw_facial_regions(frame, eyes, mouth):
    if eyes and 'pupils' in eyes and eyes['pupils']:
        for (px, py, area) in eyes['pupils']:
            radius = max(2, int(np.sqrt(area / np.pi)))
            cv2.circle(frame, (px, py), radius, (0, 0, 255), -1)
            cv2.circle(frame, (px, py), radius + 2, (0, 255, 255), 2)

def build_face_detector(device):
    try:
        gpu_id = 0 if device.type == "cuda" else -1
        return TorchRetinaFace(gpu_id=gpu_id)
    except:
        return None

def detect_faces(detector, frame):
    if detector is None:
        return []
    try:
        h, w = frame.shape[:2]
        scale = min(1.0, 480 / float(w))
        small = cv2.resize(frame, (int(w*scale), int(h*scale))) if scale < 1.0 else frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        dets = detector(rgb)
        boxes = []
        
        for det in dets:
            if det[0] is not None:
                x1, y1, x2, y2 = det[0]
                box = (int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale))
                area = (box[2] - box[0]) * (box[3] - box[1])
                
                if area < 5000:
                    continue
                
                boxes.append(box)
        
        if len(boxes) > 1:
            boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            boxes = boxes[:1]
        
        return boxes
    except Exception as e:
        return []


def main():
    emotion_model, classes = load_emotion_model(EMOTION_MODEL_PATH, DEVICE)
    detector = build_face_detector(DEVICE)
    analyzer = AdvancedFacialAnalyzer()
    calculator = StressCalculator()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("[INFO] Ready - Press 'Q' to quit")
    
    frame_count, prev_time, last_boxes = 0, time.time(), []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            if frame_count % DETECT_EVERY_N_FRAMES == 0:
                last_boxes = detect_faces(detector, frame)
            
            if not last_boxes:
                cv2.putText(frame, "No face", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            else:
                for x1, y1, x2, y2 in last_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    
                    emotion, conf = predict_emotion(crop, emotion_model, classes, DEVICE)
                    
                    eyes = analyzer.analyze_eyes(frame, (x1, y1, x2, y2))
                    mouth = analyzer.analyze_mouth(frame, (x1, y1, x2, y2))
                    
                    score, reason, weight = calculator.calculate(emotion, conf, eyes, mouth)
                    
                    if score > 70:
                        status, color = "STRESS", (0, 0, 255)
                    elif score > 50:
                        status, color = "UNCERTAIN", (0, 165, 255)
                    else:
                        status, color = "NO STRESS", (0, 255, 0)
                    
                    draw_facial_regions(frame, eyes, mouth)
                    
                    cv2.putText(frame, status, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)
                    cv2.putText(frame, f"Score: {score:.0f}%", (x1, y2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    y_info = y1 - 10
                    if eyes and eyes['closed_time'] > 0:
                        cv2.putText(frame, f"Eyes: {eyes['closed_time']:.1f}s | BR: {eyes['brightness']:.0f}", (x1, y_info), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if eyes['stress'] else (0, 255, 0), 1)
                        y_info -= 20
                    
                    cv2.putText(frame, f"M:{emotion.upper()} ({conf*100:.0f}%)", (x1, y_info), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    y_info -= 20
                    
                    if mouth:
                        cv2.putText(frame, f"MO:{mouth.get('width', 0):.0f} | S:{mouth.get('smile', False)}", (x1, y_info),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
            
            fps = 1.0 / max(time.time() - prev_time, 1e-6)
            prev_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow("AI Stress Monitor - Laptop Camera", frame)
            
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break
    
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Done")

if __name__ == "__main__":
    main()
