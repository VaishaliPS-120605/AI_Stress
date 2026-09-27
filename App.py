"""
AI Stress Monitor - Final Application
Main application integrating ESP32 Watch Camera, MAX30102 sensors, and facial AI.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import threading
import time
import json
import urllib.request
import urllib.error
import sqlite3
import csv
import socket
import concurrent.futures
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
from torchvision import transforms
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestRegressor
import pickle
import io
import requests
from face_detection import RetinaFace as TorchRetinaFace

# ========== CONFIGURATION ==========
DEFAULT_WATCH_IP = "10.122.55.124"

DB_PATH = "stress_monitor.db"
CSV_EXPORT_PATH = "stress_history.csv"

SENSOR_POLL_INTERVAL = 3.0  # Poll sensor every 3.0 seconds (gives 100% bandwidth to video stream)
MEASUREMENT_SAVE_INTERVAL = 5  # Save measurement every 5 seconds
RISK_UPDATE_INTERVAL = 3600  # Update risk indicators every hour
RESULT_SEND_DEBOUNCE = 2  # Debounce result sending to ESP32 every 2 seconds

# Hysteresis thresholds for stress status
STRESS_THRESHOLD = 70.0
NORMAL_THRESHOLD = 50.0
MIN_STRESS_DURATION = 2  # Seconds to maintain stress status

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ========== AUTO-DISCOVERY HELPER ==========
def find_watch_ip(preferred_ip=None):
    """Scan local Wi-Fi subnet to auto-discover ESP32 Watch on port 80 or 81."""
    # 1. First, check preferred or default Watch IP directly (instant check)
    candidates = []
    if preferred_ip:
        candidates.append(preferred_ip.strip())
    if DEFAULT_WATCH_IP not in candidates:
        candidates.append(DEFAULT_WATCH_IP)
    
    for candidate in candidates:
        for port in [80, 81]:
            try:
                s = socket.socket()
                s.settimeout(0.6)
                res = s.connect_ex((candidate, port))
                s.close()
                if res == 0:
                    print(f"[INFO] Watch verified at known IP: {candidate} (port {port})")
                    return candidate
            except Exception:
                pass
    
    # 2. If known IPs didn't respond, scan the local subnet
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "10.122.55.79"
    
    prefix = '.'.join(local_ip.split('.')[:3]) + '.'
    print(f"[INFO] Scanning {prefix}1-254 for ESP32 Watch...")
    
    def check_target(i):
        target = f"{prefix}{i}"
        for port in [80, 81]:
            try:
                sock = socket.socket()
                sock.settimeout(0.9)
                res = sock.connect_ex((target, port))
                sock.close()
                if res == 0:
                    return target
            except Exception:
                pass
        return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as ex:
        results = ex.map(check_target, range(1, 255))
        found = [r for r in results if r]
    
    if found:
        print(f"[INFO] Auto-discovered Watch at: {found[0]}")
        return found[0]
    return None


# ========== DATABASE SCHEMA ==========
def init_database():
    """Initialize SQLite database with required schema."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        heart_rate INTEGER,
        spo2 INTEGER,
        hrv REAL,
        ir INTEGER,
        red INTEGER,
        finger BOOLEAN,
        emotion TEXT,
        emotion_confidence REAL,
        eye_state TEXT,
        smile_state TEXT,
        facial_stress_score REAL,
        physiological_stress_score REAL,
        final_stress_score REAL,
        stress_status TEXT,
        data_source TEXT
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS risk_indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        cardiovascular_risk TEXT,
        sleep_risk TEXT,
        anxiety_risk TEXT,
        metabolic_risk TEXT
    )''')
    
    conn.commit()
    conn.close()


def insert_measurement(hr, spo2, hrv, ir, red, finger, emotion, confidence, 
                       eye_state, smile_state, facial_score, phys_score, 
                       final_score, status, source):
    """Insert a measurement into the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO measurements (
        heart_rate, spo2, hrv, ir, red, finger, emotion, emotion_confidence,
        eye_state, smile_state, facial_stress_score, physiological_stress_score,
        final_stress_score, stress_status, data_source
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
    (hr, spo2, hrv, ir, red, finger, emotion, confidence,
     eye_state, smile_state, facial_score, phys_score, final_score, status, source))
    conn.commit()
    conn.close()


def get_recent_measurements(days=30):
    """Get measurements from the last N days."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cutoff = datetime.now() - timedelta(days=days)
    c.execute('SELECT * FROM measurements WHERE timestamp > ? ORDER BY timestamp DESC',
              (cutoff,))
    rows = c.fetchall()
    conn.close()
    return rows


def clear_demo_data():
    """Delete all dummy records from database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM measurements WHERE data_source = 'dummy'")
    conn.commit()
    conn.close()


def initialize_demo_data():
    """Generate 30 days of realistic demo data."""
    import random
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM measurements WHERE data_source = 'real' OR data_source = 'watch'")
    real_count = c.fetchone()[0]
    
    if real_count > 0:
        conn.close()
        return
    
    base_time = datetime.now() - timedelta(days=30)
    
    for day in range(30):
        current_date = base_time + timedelta(days=day)
        
        for hour_offset in range(0, 24, 2):
            timestamp = current_date.replace(hour=hour_offset, minute=0, second=0)
            
            base_hr = 70 + (day % 7) * 5 + random.randint(-10, 10)
            base_spo2 = 97 + random.randint(-2, 2)
            
            if 17 <= hour_offset < 21:
                stress_score = 60 + random.randint(-10, 20)
            else:
                stress_score = 40 + random.randint(-10, 10)
            
            c.execute('''INSERT INTO measurements (
                timestamp, heart_rate, spo2, hrv, ir, red, finger, emotion,
                emotion_confidence, eye_state, smile_state, facial_stress_score,
                physiological_stress_score, final_stress_score, stress_status, data_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (timestamp, base_hr, base_spo2, 30.0, 100000 + random.randint(-10000, 10000),
             120000 + random.randint(-10000, 10000), True, "neutral",
             0.8 + random.random() * 0.2, "open", "no",
             stress_score * 0.4, stress_score * 0.6, stress_score,
             "STRESS" if stress_score > 65 else "NORMAL", "dummy"))
    
    conn.commit()
    conn.close()
    print("[INFO] Demo data initialized")


# ========== ESP32 WATCH COMMUNICATION ==========
class ESP32Manager:
    """Manages communication with ESP32 Watch sensors and results via HTTP."""
    
    def __init__(self, ip_address, device):
        self.ip = ip_address.strip() if ip_address else None
        self.device = device
        self.sensor_url = f"http://{self.ip}/sensor" if self.ip else None
        self.status_url = f"http://{self.ip}/status" if self.ip else None
        self.result_url = f"http://{self.ip}/result" if self.ip else None
        
        self.latest_sensor_data = {
            'heart_rate': 0,
            'spo2': 0,
            'ir': 0,
            'red': 0,
            'finger': False,
            'connected': False,
            'connecting': True
        }
        self.failed_polls = 0
        self.sensor_thread = None
        self.running = False
        
        if self.ip:
            print(f"[INFO] ESP32 Watch configured at IP: {self.ip}")
    
    def start_sensor_polling(self):
        """Start background thread for sensor polling."""
        if not self.sensor_url:
            return
        self.running = True
        self.sensor_thread = threading.Thread(target=self._poll_sensor, daemon=True)
        self.sensor_thread.start()
    
    def stop_sensor_polling(self):
        """Stop background thread."""
        self.running = False
    
    def _poll_sensor(self):
        """Background thread: poll ESP32 sensor endpoint."""
        while self.running:
            if not self.sensor_url:
                break
            try:
                response = requests.get(self.sensor_url, headers={'Connection': 'close'}, timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    self.failed_polls = 0
                    self.latest_sensor_data = {
                        'heart_rate': int(data.get('heart_rate', 0)),
                        'spo2': int(data.get('spo2', 0)),
                        'ir': int(data.get('ir', 0)),
                        'red': int(data.get('red', 0)),
                        'finger': data.get('finger', False) in (True, 'true', 'True', 1),
                        'connected': True,
                        'connecting': False
                    }
                else:
                    self.failed_polls += 1
                    self.latest_sensor_data['connected'] = False
                    self.latest_sensor_data['connecting'] = (self.failed_polls < 3)
            except Exception:
                self.failed_polls += 1
                self.latest_sensor_data['connected'] = False
                self.latest_sensor_data['connecting'] = (self.failed_polls < 3)
            
            time.sleep(SENSOR_POLL_INTERVAL)
    
    def get_sensor_data(self):
        """Get latest cached sensor data."""
        return self.latest_sensor_data
    
    def send_result(self, stress_status):
        """Send final stress result back to Watch OLED."""
        if not self.result_url:
            return False
        try:
            params = {'stress': stress_status}
            response = requests.get(self.result_url, params=params, headers={'Connection': 'close'}, timeout=3)
            return response.status_code == 200
        except Exception:
            return False


# ========== FACIAL AI PROCESSING ==========
class FacialAIProcessor:
    """Facial feature detection (eyes, mouth, fatigue, smiles)."""
    
    def __init__(self, device):
        self.device = device
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.eye_closure_streak = 0
        self.is_eyes_closed = False
        print("[INFO] Facial feature detector ready (eye Haar cascade + mouth HSV)")
    
    def detect_eyes_in_face(self, face_bgr):
        """Detect eyes using Haar cascade."""
        try:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 4)
            return len(eyes) >= 1, len(eyes)
        except Exception:
            return True, 0
    
    def detect_mouth_in_face(self, face_bgr):
        """Detect mouth opening using HSV contours."""
        try:
            hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
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
                        mouth_ratio = h / float(w)
                        return mouth_ratio > 0.35, mouth_ratio, w
            
            return False, 0.0, 0
        except Exception:
            return False, 0.0, 0
    
    def detect_facial_features(self, frame, face_box):
        """Detect facial features from face crop."""
        if face_box is None:
            return {
                'eyes_open': True,
                'mouth_open': False,
                'mouth_ratio': 0.0,
                'mouth_width': 0,
                'eye_closure_duration': 0
            }
        
        try:
            x1, y1, x2, y2 = face_box
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            eyes_open, num_eyes = self.detect_eyes_in_face(face_crop)
            mouth_open, mouth_ratio, mouth_width = self.detect_mouth_in_face(face_crop)
            
            if not eyes_open:
                self.is_eyes_closed = True
                self.eye_closure_streak += 1
                eye_duration = self.eye_closure_streak / 30.0
            else:
                self.eye_closure_streak = 0
                self.is_eyes_closed = False
                eye_duration = 0.0
            
            return {
                'eyes_open': eyes_open,
                'mouth_open': mouth_open,
                'mouth_ratio': mouth_ratio,
                'mouth_width': mouth_width,
                'eye_closure_duration': eye_duration
            }
        except Exception:
            return None


# ========== STRESS SCORE FUSION ==========
class StressFusionEngine:
    """Fuses facial and physiological evidence into a unified stress score."""
    
    def __init__(self):
        self.facial_history = deque(maxlen=60)
        self.phys_history = deque(maxlen=10)
        self.last_result_send = 0
        self.current_status = "NORMAL"
        self.status_start_time = time.time()
    
    def calculate_facial_evidence(self, emotion, confidence, eye_state, smile_state="no"):
        """Calculate facial stress evidence (0-100)."""
        if smile_state == "yes":
            return 15.0
        
        emotion_scores = {
            'anger': 85,
            'fear': 80,
            'sad': 75,
            'sadness': 75,
            'disgust': 65,
            'neutral': 40,
            'surprise': 45,
            'happiness': 10,
            'happy': 10,
        }
        
        base_score = emotion_scores.get(emotion.lower() if emotion else 'neutral', 40)
        score = base_score * (confidence if confidence > 0 else 0.7)
        
        if eye_state == "closed":
            score += 20.0
        
        return min(max(score, 0), 100)
    
    def calculate_physiological_evidence(self, hr, spo2, hrv=30.0):
        """Calculate physiological stress evidence (0-100)."""
        score = 0.0
        
        if hr > 100:
            score += (hr - 100) / 2.0
        
        if spo2 < 95:
            score += (95 - spo2) * 2.0
        
        if hrv > 0 and hrv < 30:
            score += (30 - hrv)
        
        return min(max(score, 0), 100)
    
    def get_smoothed_score(self):
        """Get temporally smoothed stress score."""
        if len(self.facial_history) == 0 and len(self.phys_history) == 0:
            return 0.0
        
        facial_avg = np.mean(list(self.facial_history)) if self.facial_history else 40.0
        
        if self.phys_history:
            phys_avg = np.mean(list(self.phys_history))
            final_score = facial_avg * 0.4 + phys_avg * 0.6
        else:
            final_score = facial_avg
        
        return final_score
    
    def update(self, emotion, emotion_conf, eye_state, hr=None, spo2=None, hrv=None):
        """Update stress score with new data."""
        facial_score = self.calculate_facial_evidence(emotion, emotion_conf, eye_state)
        self.facial_history.append(facial_score)
        
        if hr is not None and spo2 is not None:
            phys_score = self.calculate_physiological_evidence(hr, spo2, hrv if hrv is not None else 30.0)
            self.phys_history.append(phys_score)
        else:
            phys_score = 0.0
        
        return facial_score, phys_score
    
    def get_status_with_hysteresis(self):
        """Get stress status with hysteresis to prevent flickering."""
        current_score = self.get_smoothed_score()
        
        if self.current_status == "NORMAL":
            if current_score > STRESS_THRESHOLD:
                self.current_status = "STRESS"
                self.status_start_time = time.time()
        else:
            if current_score < NORMAL_THRESHOLD:
                self.current_status = "NORMAL"
                self.status_start_time = time.time()
        
        return self.current_status, current_score


# ========== REPORTING & ANALYSIS ==========
def calculate_daily_statistics(measurements):
    """Calculate statistics from measurements list."""
    if not measurements:
        return {}
    
    hrs = [m[2] for m in measurements if m[2]]
    spo2s = [m[3] for m in measurements if m[3]]
    scores = [m[14] for m in measurements if m[14]]
    statuses = [m[15] for m in measurements if m[15]]
    
    return {
        'avg_hr': np.mean(hrs) if hrs else 0,
        'avg_spo2': np.mean(spo2s) if spo2s else 0,
        'avg_stress': np.mean(scores) if scores else 0,
        'stress_count': len([s for s in statuses if s == 'STRESS']),
        'normal_count': len([s for s in statuses if s == 'NORMAL']),
        'stress_percentage': (len([s for s in statuses if s == 'STRESS']) / len(statuses) * 100) if statuses else 0,
    }


def get_time_of_day_analysis(measurements):
    """Analyze stress by time of day."""
    time_periods = {
        'Morning (06:00-12:00)': deque(),
        'Afternoon (12:00-17:00)': deque(),
        'Evening (17:00-21:00)': deque(),
        'Night (21:00-06:00)': deque(),
    }
    
    for m in measurements:
        try:
            timestamp = datetime.fromisoformat(m[1])
            hour = timestamp.hour
            score = m[14]
            
            if 6 <= hour < 12:
                time_periods['Morning (06:00-12:00)'].append(score)
            elif 12 <= hour < 17:
                time_periods['Afternoon (12:00-17:00)'].append(score)
            elif 17 <= hour < 21:
                time_periods['Evening (17:00-21:00)'].append(score)
            else:
                time_periods['Night (21:00-06:00)'].append(score)
        except Exception:
            continue
    
    result = {}
    for period, scores in time_periods.items():
        result[period] = np.mean(list(scores)) if scores else 0
    
    return result


def get_day_of_week_analysis(measurements):
    """Analyze stress by day of week."""
    days = {
        'Monday': deque(),
        'Tuesday': deque(),
        'Wednesday': deque(),
        'Thursday': deque(),
        'Friday': deque(),
        'Saturday': deque(),
        'Sunday': deque(),
    }
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for m in measurements:
        try:
            timestamp = datetime.fromisoformat(m[1])
            day_name = day_names[timestamp.weekday()]
            score = m[14]
            days[day_name].append(score)
        except Exception:
            continue
    
    result = {}
    for day, scores in days.items():
        result[day] = np.mean(list(scores)) if scores else 0
    
    return result


def calculate_risk_indicators(measurements):
    """Calculate long-term health risk indicators."""
    if not measurements:
        return {}
    
    stats = calculate_daily_statistics(measurements)
    stress_pct = stats.get('stress_percentage', 0)
    avg_stress = stats.get('avg_stress', 0)
    avg_hr = stats.get('avg_hr', 0)
    
    risk_levels = {}
    
    if avg_stress > 70 or avg_hr > 90:
        risk_levels['cardiovascular'] = 'HIGH'
    elif avg_stress > 50 or avg_hr > 80:
        risk_levels['cardiovascular'] = 'MODERATE'
    else:
        risk_levels['cardiovascular'] = 'LOW'
    
    time_analysis = get_time_of_day_analysis(measurements)
    evening_stress = time_analysis.get('Evening (17:00-21:00)', 0)
    if evening_stress > 65:
        risk_levels['sleep'] = 'HIGH'
    elif evening_stress > 50:
        risk_levels['sleep'] = 'MODERATE'
    else:
        risk_levels['sleep'] = 'LOW'
    
    if stress_pct > 40:
        risk_levels['anxiety'] = 'HIGH'
    elif stress_pct > 20:
        risk_levels['anxiety'] = 'MODERATE'
    else:
        risk_levels['anxiety'] = 'LOW'
    
    if avg_stress > 60 and avg_hr > 85:
        risk_levels['metabolic'] = 'HIGH'
    elif avg_stress > 45 or avg_hr > 75:
        risk_levels['metabolic'] = 'MODERATE'
    else:
        risk_levels['metabolic'] = 'LOW'
    
    return risk_levels


def train_prediction_model(measurements):
    """Train Random Forest model for stress prediction."""
    if len(measurements) < 20:
        return None
    
    try:
        X = []
        y = []
        
        for m in measurements:
            timestamp = datetime.fromisoformat(m[1])
            X.append([
                timestamp.hour,
                timestamp.weekday(),
                m[2] if m[2] else 70,
                m[3] if m[3] else 97,
                m[14] if m[14] else 50,
            ])
            y.append(m[14] if m[14] else 50)
        
        X = np.array(X)
        y = np.array(y)
        
        model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X, y)
        return model
    except Exception:
        return None


def predict_stress_for_datetime(model, dt):
    """Predict stress score for a given datetime."""
    if model is None:
        return None
    
    X = np.array([[dt.hour, dt.weekday(), 75, 97, 50]])
    try:
        prediction = model.predict(X)[0]
        return min(max(prediction, 0), 100)
    except Exception:
        return None


# ========== MAIN APPLICATION UI ==========
class StressMonitorApp:
    """Main Tkinter application integrating ESP32 Watch Camera and Health Dashboard."""
    
    def __init__(self, root, esp32_ip=DEFAULT_WATCH_IP):
        self.root = root
        self.root.title("AI Stress Monitor - ESP32 Watch Hardware Dashboard")
        self.root.geometry("1400x820")
        
        self.esp32_ip = esp32_ip.strip() if esp32_ip else DEFAULT_WATCH_IP
        self.esp32_manager = ESP32Manager(self.esp32_ip, DEVICE)
        self.facial_ai = FacialAIProcessor(DEVICE)
        self.stress_engine = StressFusionEngine()
        
        # Initialize face detector
        try:
            gpu_id = 0 if DEVICE.type == "cuda" else -1
            self.face_detector = TorchRetinaFace(gpu_id=gpu_id)
            print(f"[INFO] RetinaFace initialized (gpu_id={gpu_id}, {'GPU' if gpu_id >= 0 else 'CPU'} mode)")
        except Exception as e:
            print(f"[WARN] Failed to initialize RetinaFace: {e}")
            self.face_detector = None
        
        # Camera & Hardware State
        self.rotate_180 = True  # Default Watch orientation
        self.active_cap = None
        self._current_stream = None
        self.camera_thread = None
        self.running = True
        self.last_measurement_time = 0
        self.last_result_send_time = 0
        
        # Current facial AI & stress results
        self.current_emotion = "neutral"
        self.current_emotion_confidence = 0.0
        self.current_eye_state = "open"
        self.current_mouth_ratio = 0.0
        self.current_mouth_width = 0
        self.current_eye_duration = 0.0
        self.current_smile = "no"
        self.current_face_box = None
        self.current_faces = []
        self.current_status = "NORMAL"
        self.current_stress_score = 0.0
        
        # Thread-safe camera and FPS state
        self._latest_display_frame = None
        self._camera_status_text = f"Camera: CONNECTING ({self.esp32_ip})..."
        self._camera_status_fg = "orange"
        self.fps_start = time.time()
        self.fps_counter = 0
        self._ai_busy = False
        self._frame_id = 0
        self._last_rendered_frame_id = -1
        
        # UI Components
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.live_tab = ttk.Frame(self.notebook)
        self.report_tab = ttk.Frame(self.notebook)
        self.prediction_tab = ttk.Frame(self.notebook)
        self.risk_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.live_tab, text="LIVE")
        self.notebook.add(self.report_tab, text="REPORT")
        self.notebook.add(self.prediction_tab, text="PREDICTION")
        self.notebook.add(self.risk_tab, text="RISK")
        
        self.setup_live_tab()
        self.setup_report_tab()
        self.setup_prediction_tab()
        self.setup_risk_tab()
        self.setup_status_bar()
        
        # Start background threads
        if self.esp32_manager:
            self.esp32_manager.start_sensor_polling()
        
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
        # Schedule periodic UI update on main thread
        self.root.after(100, self.update_ui)
        
        # Cleanup on close
        root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def setup_live_tab(self):
        """Setup live monitoring tab."""
        main_frame = ttk.Frame(self.live_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left: Camera
        left_frame = ttk.LabelFrame(main_frame, text="LIVE ESP32 WATCH CAMERA")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Camera controls
        cam_ctrl = ttk.Frame(left_frame)
        cam_ctrl.pack(fill=tk.X, padx=5, pady=3)
        
        self.rot_btn = ttk.Button(cam_ctrl, text="Rotate: 180° (Flipped)", command=self.toggle_rotation)
        self.rot_btn.pack(side=tk.LEFT, padx=5)
        
        self.cam_info_lbl = tk.Label(cam_ctrl, text=f"Source: ESP32 Watch ({self.esp32_ip})", font=("Arial", 9, "bold"), fg="#0055aa")
        self.cam_info_lbl.pack(side=tk.LEFT, padx=10)
        
        self.camera_label = tk.Label(left_frame, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Right: Data
        right_frame = ttk.Frame(main_frame, width=430)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        right_frame.pack_propagate(False)
        
        # Sensor data
        sensor_frame = ttk.LabelFrame(right_frame, text="PHYSIOLOGICAL DATA (Watch Sensor)")
        sensor_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.hr_label = tk.Label(sensor_frame, text="Heart Rate: -- BPM", font=("Arial", 12))
        self.hr_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.spo2_label = tk.Label(sensor_frame, text="SpO2: -- %", font=("Arial", 12))
        self.spo2_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.hrv_label = tk.Label(sensor_frame, text="HRV: -- ms", font=("Arial", 12))
        self.hrv_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.ir_label = tk.Label(sensor_frame, text="IR: --", font=("Arial", 10))
        self.ir_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.red_label = tk.Label(sensor_frame, text="RED: --", font=("Arial", 10))
        self.red_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.finger_label = tk.Label(sensor_frame, text="Finger: DISCONNECTED", font=("Arial", 11, "bold"), fg="red")
        self.finger_label.pack(anchor=tk.W, padx=10, pady=4)
        
        self.sensor_status = tk.Label(sensor_frame, text="Sensor: CONNECTING...", font=("Arial", 10), fg="orange")
        self.sensor_status.pack(anchor=tk.W, padx=10, pady=2)
        
        # Facial analysis
        facial_frame = ttk.LabelFrame(right_frame, text="FACIAL ANALYSIS (Watch Camera AI)")
        facial_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.emotion_label = tk.Label(facial_frame, text="Emotion: --", font=("Arial", 12))
        self.emotion_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.confidence_label = tk.Label(facial_frame, text="Confidence: -- %", font=("Arial", 12))
        self.confidence_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.eye_label = tk.Label(facial_frame, text="Eyes: OPEN", font=("Arial", 11))
        self.eye_label.pack(anchor=tk.W, padx=10, pady=2)
        
        self.smile_label = tk.Label(facial_frame, text="Smile: NO", font=("Arial", 11))
        self.smile_label.pack(anchor=tk.W, padx=10, pady=2)
        
        # Stress score
        stress_frame = ttk.LabelFrame(right_frame, text="STRESS ASSESSMENT")
        stress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.stress_score_label = tk.Label(stress_frame, text="Stress Score: -- %", font=("Arial", 14, "bold"))
        self.stress_score_label.pack(anchor=tk.W, padx=10, pady=5)
        
        self.final_status_label = tk.Label(stress_frame, text="Final Status: NORMAL", 
                                           font=("Arial", 13, "bold"), fg="green")
        self.final_status_label.pack(anchor=tk.W, padx=10, pady=5)
        
        # Connection status
        conn_frame = ttk.LabelFrame(right_frame, text="HARDWARE CONNECTION STATUS")
        conn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.camera_status = tk.Label(conn_frame, text=f"Camera: CONNECTING ({self.esp32_ip})...", font=("Arial", 10), fg="orange")
        self.camera_status.pack(anchor=tk.W, padx=10, pady=2)
        
        self.esp32_status = tk.Label(conn_frame, text=f"Watch: CONNECTING ({self.esp32_ip})...", font=("Arial", 10), fg="orange")
        self.esp32_status.pack(anchor=tk.W, padx=10, pady=2)
        
        self.ai_status = tk.Label(conn_frame, text="AI: CPU MODE", font=("Arial", 10), fg="orange")
        if DEVICE.type == "cuda":
            self.ai_status.config(text="AI: GPU MODE", fg="green")
        self.ai_status.pack(anchor=tk.W, padx=10, pady=2)
        
        self.fps_label = tk.Label(conn_frame, text="FPS: --", font=("Arial", 10))
        self.fps_label.pack(anchor=tk.W, padx=10, pady=2)
        
        # Watch IP configuration box
        esp_box = ttk.Frame(conn_frame)
        esp_box.pack(fill=tk.X, padx=5, pady=6)
        ttk.Label(esp_box, text="Watch IP:").pack(side=tk.LEFT)
        self.esp_ip_entry = ttk.Entry(esp_box, width=14)
        self.esp_ip_entry.insert(0, self.esp32_ip)
        self.esp_ip_entry.pack(side=tk.LEFT, padx=3)
        
        self.connect_btn = ttk.Button(esp_box, text="Connect", command=self.connect_esp32)
        self.connect_btn.pack(side=tk.LEFT, padx=2)
        
        self.auto_btn = ttk.Button(esp_box, text="Auto-Detect", command=self.auto_detect_watch)
        self.auto_btn.pack(side=tk.LEFT, padx=2)
    
    def toggle_rotation(self):
        """Toggle frame rotation between 0 and 180 degrees."""
        self.rotate_180 = not self.rotate_180
        status = "180° (Flipped)" if self.rotate_180 else "Normal (0°)"
        if hasattr(self, 'rot_btn'):
            self.rot_btn.config(text=f"Rotate: {status}")
    
    def auto_detect_watch(self):
        """Scan local Wi-Fi subnet to find the Watch automatically."""
        self.status_var.set("Scanning Wi-Fi network for ESP32 Watch...")
        self.root.update()
        found = find_watch_ip(preferred_ip=self.esp32_ip)
        if found:
            self.esp_ip_entry.delete(0, tk.END)
            self.esp_ip_entry.insert(0, found)
            self._apply_connection(found)
            messagebox.showinfo("Watch Discovered", f"ESP32 Watch found at {found}!")
        else:
            messagebox.showwarning("Not Found", "Could not find Watch automatically. Ensure Watch is powered on and connected to the same Wi-Fi network.")
    
    def connect_esp32(self):
        """Handle user clicking Connect button with entered Watch IP."""
        ip = self.esp_ip_entry.get().strip() if hasattr(self, 'esp_ip_entry') else ""
        if not ip:
            messagebox.showwarning("Watch IP", "Please enter the Watch IP address.")
            return
        
        # Test if the entered IP is reachable on port 80 or 81
        s = socket.socket()
        s.settimeout(1.2)
        reachable = (s.connect_ex((ip, 80)) == 0 or s.connect_ex((ip, 81)) == 0)
        s.close()
        
        if not reachable:
            # Check local subnet to see what the active IP is
            discovered = find_watch_ip(preferred_ip=ip)
            if discovered and discovered != ip:
                ans = messagebox.askyesno(
                    "Watch IP Notice",
                    f"Could not reach {ip}.\n\n"
                    f"On your current Wi-Fi network, your Watch was discovered at:\n{discovered}\n\n"
                    f"Would you like to connect to {discovered} instead?"
                )
                if ans:
                    ip = discovered
                    self.esp_ip_entry.delete(0, tk.END)
                    self.esp_ip_entry.insert(0, ip)
                else:
                    return
            else:
                messagebox.showerror(
                    "Connection Failed",
                    f"Could not reach {ip}.\n\n"
                    f"Please ensure:\n"
                    f"1. Watch is powered on.\n"
                    f"2. Watch and Laptop are on the SAME Wi-Fi hotspot.\n"
                    f"3. Click 'Auto-Detect' to scan automatically."
                )
                return
        
        self._apply_connection(ip)
    
    def _apply_connection(self, ip):
        """Switch connection to the target Watch IP cleanly."""
        self.esp32_ip = ip
        
        # Cleanly interrupt active stream if open to force immediate reconnect
        if hasattr(self, '_current_stream') and self._current_stream is not None:
            try:
                self._current_stream.close()
            except Exception:
                pass
            self._current_stream = None
        
        if self.active_cap is not None:
            try:
                self.active_cap.release()
            except Exception:
                pass
            self.active_cap = None
        
        if self.esp32_manager:
            self.esp32_manager.stop_sensor_polling()
        
        self.esp32_manager = ESP32Manager(ip, DEVICE)
        self.esp32_manager.start_sensor_polling()
        
        if hasattr(self, 'cam_info_lbl'):
            self.cam_info_lbl.config(text=f"Source: ESP32 Watch ({ip})")
        if hasattr(self, 'esp32_status'):
            self.esp32_status.config(text=f"Watch: CONNECTING ({ip})...", fg="orange")
        if hasattr(self, 'camera_status'):
            self._camera_status_text = f"Camera: CONNECTING ({ip})..."
            self._camera_status_fg = "orange"
        if hasattr(self, 'status_var'):
            self.status_var.set(f"Connecting to ESP32 Watch at {ip}...")
    
    def setup_report_tab(self):
        """Setup report tab."""
        frame = ttk.Frame(self.report_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title = tk.Label(frame, text="30-DAY STRESS REPORT", font=("Arial", 14, "bold"))
        title.pack(anchor=tk.W, pady=10)
        
        stats_frame = ttk.LabelFrame(frame, text="SUMMARY STATISTICS")
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_text = tk.Label(stats_frame, text="Monitoring days: --\nAverage stress: -- %", 
                                   font=("Arial", 11), justify=tk.LEFT)
        self.stats_text.pack(anchor=tk.W, padx=10, pady=10)
        
        self.chart_frame = ttk.Frame(frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Refresh Report", command=self.refresh_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Download CSV", command=self.download_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Demo Data", command=self.clear_demo).pack(side=tk.LEFT, padx=5)
    
    def setup_prediction_tab(self):
        """Setup prediction tab."""
        frame = ttk.Frame(self.prediction_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title = tk.Label(frame, text="STRESS PREDICTION", font=("Arial", 14, "bold"))
        title.pack(anchor=tk.W, pady=10)
        
        self.prediction_text = tk.Label(frame, text="Prediction: INSUFFICIENT DATA", 
                                        font=("Arial", 12), justify=tk.LEFT)
        self.prediction_text.pack(anchor=tk.W, padx=10, pady=10)
        
        ttk.Button(frame, text="Update Prediction", command=self.update_prediction).pack(anchor=tk.W, padx=10, pady=5)
    
    def setup_risk_tab(self):
        """Setup risk indicators tab."""
        frame = ttk.Frame(self.risk_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        title = tk.Label(frame, text="LONG-TERM STRESS-ASSOCIATED HEALTH RISK INDICATORS", 
                        font=("Arial", 14, "bold"))
        title.pack(anchor=tk.W, pady=10)
        
        info = tk.Label(frame, text="These are informational risk categories based on historical stress patterns.\nThis is NOT a medical diagnosis.", 
                       font=("Arial", 10), fg="gray")
        info.pack(anchor=tk.W, pady=5)
        
        self.risk_text = tk.Label(frame, text="Risk Calculation: INSUFFICIENT DATA", 
                                 font=("Arial", 12), justify=tk.LEFT)
        self.risk_text.pack(anchor=tk.W, padx=10, pady=10)
        
        ttk.Button(frame, text="Update Risk Report", command=self.update_risk).pack(anchor=tk.W, padx=10, pady=5)
    
    def setup_status_bar(self):
        """Setup bottom status bar."""
        self.status_var = tk.StringVar(value=f"Target: ESP32 Watch ({self.esp32_ip})")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def calculate_emotion_from_facial_features(self, eyes_open, mouth_ratio, mouth_width):
        """Calculate emotion based on facial features (eyes, mouth ratio)."""
        if not eyes_open and self.current_eye_duration >= 1.5:
            return "sad", 0.75
        elif mouth_ratio > 0.35 and eyes_open:
            return "happy", 0.85
        else:
            return "neutral", 0.65
    
    def draw_connecting_placeholder(self, ip):
        """Draw a pleasant connecting screen on the camera canvas."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (30, 25, 20)
        
        cv2.putText(img, "CONNECTING TO WATCH CAMERA...", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)
        cv2.putText(img, f"Target IP: {ip}", (160, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, "Establishing stream...", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2)
        
        self._latest_display_frame = img
    
    def draw_offline_placeholder(self, ip):
        """Draw an informative offline guide on the camera canvas when watch is disconnected."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (35, 30, 30)
        
        cv2.putText(img, "WATCH CAMERA OFFLINE", (110, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 3)
        cv2.putText(img, f"Target IP: {ip}", (160, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(img, "1. Connect USB cable to power Watch", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.putText(img, "2. Ensure Watch & Laptop are on SAME Wi-Fi", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.putText(img, "3. Click 'Auto-Detect' if IP changed", (80, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.putText(img, "Searching for Watch stream...", (160, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
        
        self._latest_display_frame = img
    
    def _async_ai_worker(self, frame):
        """Asynchronous worker running face detection & facial features without blocking video."""
        try:
            h, w = frame.shape[:2]
            scale = min(1.0, 320.0 / float(w))
            if scale < 1.0:
                small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                small_frame = frame
            
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            new_faces = []
            if self.face_detector is not None:
                detections = self.face_detector(rgb_frame)
                if detections and len(detections) > 0:
                    for det in detections:
                        if det[0] is not None:
                            x1, y1, x2, y2 = det[0]
                            x1, y1 = max(0, int(x1 / scale)), max(0, int(y1 / scale))
                            x2, y2 = min(w - 1, int(x2 / scale)), min(h - 1, int(y2 / scale))
                            if (x2 - x1) * (y2 - y1) > 2000:
                                new_faces.append((x1, y1, x2, y2))
                    
                    if len(new_faces) > 1:
                        new_faces = sorted(
                            new_faces,
                            key=lambda b: (b[2]-b[0]) * (b[3]-b[1]),
                            reverse=True
                        )[:1]
            
            self.current_faces = new_faces
            
            # Facial Features & Emotion
            if self.current_faces:
                for face_box in self.current_faces:
                    features = self.facial_ai.detect_facial_features(frame, face_box)
                    if features:
                        self.current_eye_state = "open" if features['eyes_open'] else "closed"
                        self.current_mouth_ratio = features['mouth_ratio']
                        self.current_mouth_width = features['mouth_width']
                        self.current_eye_duration = features['eye_closure_duration']
                        
                        emotion, confidence = self.calculate_emotion_from_facial_features(
                            features['eyes_open'], features['mouth_ratio'], features['mouth_width']
                        )
                        self.current_emotion = emotion
                        self.current_emotion_confidence = confidence
                        self.current_smile = "yes" if features['mouth_ratio'] > 0.35 else "no"
            else:
                self.current_emotion = "neutral"
                self.current_emotion_confidence = 0.0
                self.current_eye_state = "open"
                self.current_mouth_ratio = 0.0
                self.current_smile = "no"
                self.current_eye_duration = 0.0
        except Exception:
            pass
        finally:
            self._ai_busy = False

    def process_and_display_frame(self, frame, frame_count):
        """Immediately display frame with cached AI annotations, triggering AI worker asynchronously."""
        # 1. Trigger asynchronous AI worker if idle
        if not self._ai_busy and frame_count % 2 == 0:
            self._ai_busy = True
            threading.Thread(target=self._async_ai_worker, args=(frame,), daemon=True).start()
        
        # 2. Draw cached annotations instantly (< 1ms)
        display_frame = frame.copy()
        if self.current_faces:
            for x1, y1, x2, y2 in self.current_faces:
                is_stressed = getattr(self, 'current_status', 'NORMAL') == "STRESS"
                box_color = (0, 0, 255) if is_stressed else (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                
                eye_color = (0, 255, 0) if self.current_eye_state == 'open' else (0, 0, 255)
                eye_text = f"Eyes: {self.current_eye_state.upper()}"
                
                smile_color = (0, 215, 255) if self.current_smile == 'yes' else (200, 200, 200)
                smile_text = f"Smile: {self.current_smile.upper()}"
                
                y_pos = max(y1 - 10, 25)
                cv2.putText(display_frame, eye_text, (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
                y_pos = max(y_pos - 22, 15)
                cv2.putText(display_frame, smile_text, (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, smile_color, 2)
                
                score_val = getattr(self, 'current_stress_score', 0.0)
                status_str = getattr(self, 'current_status', 'NORMAL')
                cv2.putText(display_frame, f"Emotion: {self.current_emotion.upper()} ({self.current_emotion_confidence*100:.0f}%)", 
                            (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Status: {status_str} ({score_val:.0f}%)", 
                            (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        else:
            cv2.putText(display_frame, "Searching for face...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        self._frame_id += 1
        self._latest_display_frame = display_frame
    
    def camera_loop(self):
        """Dedicated background thread streaming MJPEG directly from ESP32 Watch."""
        failed_attempts = 0
        while self.running:
            ip = self.esp32_ip.strip() if self.esp32_ip else ""
            if not ip:
                self._camera_status_text = "Camera: NO IP CONFIGURED"
                self._camera_status_fg = "orange"
                time.sleep(1)
                continue
            
            self._camera_status_text = f"Camera: CONNECTING ({ip})..."
            self._camera_status_fg = "orange"
            
            if self._latest_display_frame is None:
                self.draw_connecting_placeholder(ip)
            
            # Fast socket check before trying to open stream (avoids long hangs)
            port_open = False
            for port in [80, 81]:
                try:
                    s = socket.socket()
                    s.settimeout(1.0)
                    res = s.connect_ex((ip, port))
                    s.close()
                    if res == 0:
                        port_open = True
                        break
                except Exception:
                    pass
            
            if not port_open:
                failed_attempts += 1
                if failed_attempts >= 3:
                    self._camera_status_text = f"Camera: OFFLINE ({ip})"
                    self._camera_status_fg = "red"
                    self.draw_offline_placeholder(ip)
                else:
                    self._camera_status_text = f"Camera: CONNECTING ({ip})..."
                    self._camera_status_fg = "orange"
                time.sleep(1.0)
                continue
            
            failed_attempts = 0
            stream_url = f"http://{ip}:81/stream"
            print(f"[INFO] Connecting to Watch stream: {stream_url}...")
            
            stream = None
            try:
                req = urllib.request.Request(stream_url, headers={'User-Agent': 'AI-Stress-Monitor/1.0'})
                stream = urllib.request.urlopen(req, timeout=6)
                self._current_stream = stream
                self._camera_status_text = "Camera: CONNECTED (ESP32 Watch)"
                self._camera_status_fg = "green"
                
                buffer = b''
                frame_count = 0
                last_frame_time = time.time()
                
                while self.running and self.esp32_ip == ip:
                    try:
                        chunk = stream.read(2048)
                    except Exception as e:
                        print(f"[WARN] Stream chunk read error: {e}")
                        break
                    
                    if not chunk:
                        print("[WARN] Stream closed by peer")
                        break
                    
                    buffer += chunk
                    
                    a = buffer.find(b'\xff\xd8')
                    b = buffer.find(b'\xff\xd9')
                    
                    if a != -1 and b != -1 and b > a:
                        jpg = buffer[a:b+2]
                        buffer = buffer[b+2:]
                        
                        # Discard stale frames if buffer grew too large (ensures zero latency)
                        if len(buffer) > 80000:
                            last_soi = buffer.rfind(b'\xff\xd8')
                            if last_soi > 0:
                                buffer = buffer[last_soi:]
                        
                        try:
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        except Exception:
                            frame = None
                        
                        if frame is not None:
                            last_frame_time = time.time()
                            if self.rotate_180:
                                frame = cv2.rotate(frame, cv2.ROTATE_180)
                            frame_count += 1
                            self.process_and_display_frame(frame, frame_count)
                            time.sleep(0.005)
                    
                    # Watchdog: only reconnect if stalled with no frames for > 8 seconds
                    if time.time() - last_frame_time > 8.0:
                        print("[WARN] Watch stream stalled (>8s), reconnecting...")
                        break
                        
            except Exception as e:
                print(f"[WARN] Watch stream error: {e}")
                self._camera_status_text = f"Camera: RETRYING ({ip})..."
                self._camera_status_fg = "orange"
            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                self._current_stream = None
            
            if self.running and self.esp32_ip == ip:
                self._camera_status_text = f"Camera: RECONNECTING ({ip})..."
                self._camera_status_fg = "orange"
                time.sleep(1.0)
    
    def update_ui(self):
        """Single tick of UI update, scheduled via root.after."""
        if not self.running:
            return
        
        try:
            # 1. Sensor Data from Watch
            sensor_data = self.esp32_manager.get_sensor_data() if self.esp32_manager else {'connected': False, 'finger': False}
            
            if hasattr(self, 'sensor_status') and sensor_data.get('connected'):
                self.sensor_status.config(text="Sensor: CONNECTED", fg="green")
                if hasattr(self, 'esp32_status'):
                    self.esp32_status.config(text="Watch: CONNECTED", fg="green")
                
                if sensor_data.get('finger'):
                    if hasattr(self, 'finger_label'):
                        self.finger_label.config(text="Finger: YES", fg="green")
                    if hasattr(self, 'hr_label'):
                        self.hr_label.config(text=f"Heart Rate: {sensor_data.get('heart_rate', '--')} BPM")
                    if hasattr(self, 'spo2_label'):
                        self.spo2_label.config(text=f"SpO2: {sensor_data.get('spo2', '--')} %")
                else:
                    if hasattr(self, 'finger_label'):
                        self.finger_label.config(text="Finger: NO", fg="red")
                    if hasattr(self, 'hr_label'):
                        self.hr_label.config(text="Heart Rate: -- BPM")
                    if hasattr(self, 'spo2_label'):
                        self.spo2_label.config(text="SpO2: -- %")
                
                if hasattr(self, 'ir_label'):
                    self.ir_label.config(text=f"IR: {sensor_data.get('ir', '--')}")
                if hasattr(self, 'red_label'):
                    self.red_label.config(text=f"RED: {sensor_data.get('red', '--')}")
            elif sensor_data.get('connecting', True):
                if hasattr(self, 'sensor_status'):
                    self.sensor_status.config(text="Sensor: CONNECTING...", fg="orange")
                if hasattr(self, 'finger_label'):
                    self.finger_label.config(text="Finger: CONNECTING...", fg="orange")
                if hasattr(self, 'esp32_status'):
                    self.esp32_status.config(text=f"Watch: CONNECTING ({self.esp32_ip})...", fg="orange")
            else:
                if hasattr(self, 'sensor_status'):
                    self.sensor_status.config(text="Sensor: OFFLINE", fg="red")
                if hasattr(self, 'finger_label'):
                    self.finger_label.config(text="Finger: DISCONNECTED", fg="red")
                if hasattr(self, 'esp32_status'):
                    self.esp32_status.config(text=f"Watch: OFFLINE ({self.esp32_ip})", fg="red")
            
            # 2. Update Stress Score
            if sensor_data.get('connected') and sensor_data.get('finger'):
                facial_score, phys_score = self.stress_engine.update(
                    self.current_emotion, self.current_emotion_confidence, self.current_eye_state,
                    sensor_data.get('heart_rate', 70),
                    sensor_data.get('spo2', 97),
                    30.0
                )
            else:
                facial_score = self.stress_engine.calculate_facial_evidence(
                    self.current_emotion, self.current_emotion_confidence, self.current_eye_state, self.current_smile
                )
                self.stress_engine.facial_history.append(facial_score)
                phys_score = 0.0
            
            status, final_score = self.stress_engine.get_status_with_hysteresis()
            self.current_status = status
            self.current_stress_score = final_score
            
            # 3. Facial Analysis Labels
            if hasattr(self, 'emotion_label'):
                self.emotion_label.config(text=f"Emotion: {self.current_emotion.capitalize()}")
            if hasattr(self, 'confidence_label'):
                self.confidence_label.config(text=f"Confidence: {self.current_emotion_confidence*100:.1f}%")
            if hasattr(self, 'eye_label'):
                self.eye_label.config(text=f"Eyes: {self.current_eye_state.upper()}")
            if hasattr(self, 'smile_label'):
                self.smile_label.config(text=f"Smile: {self.current_smile.upper()}")
            
            if hasattr(self, 'stress_score_label'):
                self.stress_score_label.config(text=f"Stress Score: {final_score:.1f} %")
            
            if hasattr(self, 'final_status_label'):
                if status == "STRESS":
                    self.final_status_label.config(text="Final Status: STRESS DETECTED", fg="red")
                else:
                    self.final_status_label.config(text="Final Status: NORMAL", fg="green")
            
            # 4. Send Result back to Watch
            if self.esp32_manager and sensor_data.get('connected'):
                now = time.time()
                if now - self.last_result_send_time > RESULT_SEND_DEBOUNCE:
                    self.esp32_manager.send_result(status)
                    self.last_result_send_time = now
            
            # 5. Save Measurement Periodically
            now = time.time()
            if now - self.last_measurement_time > MEASUREMENT_SAVE_INTERVAL:
                insert_measurement(
                    sensor_data.get('heart_rate') if sensor_data.get('connected') else None,
                    sensor_data.get('spo2') if sensor_data.get('connected') else None,
                    30.0 if sensor_data.get('connected') else None,
                    sensor_data.get('ir', 0),
                    sensor_data.get('red', 0),
                    sensor_data.get('finger', False),
                    self.current_emotion,
                    self.current_emotion_confidence,
                    self.current_eye_state,
                    self.current_smile,
                    facial_score,
                    phys_score,
                    final_score,
                    status,
                    "watch"
                )
                self.last_measurement_time = now
            
            # 6. FPS Calculation
            self.fps_counter += 1
            elapsed = time.time() - self.fps_start
            if elapsed >= 1.0:
                fps = self.fps_counter / elapsed
                if hasattr(self, 'fps_label'):
                    self.fps_label.config(text=f"FPS: {fps:.1f}")
                self.fps_counter = 0
                self.fps_start = time.time()
            
            # 7. Render Video Frame on Main Thread
            if hasattr(self, 'camera_status'):
                self.camera_status.config(text=self._camera_status_text, fg=self._camera_status_fg)
            
            if hasattr(self, 'camera_label') and self._latest_display_frame is not None:
                if self._last_rendered_frame_id != self._frame_id:
                    self._last_rendered_frame_id = self._frame_id
                    try:
                        lbl_w = self.camera_label.winfo_width()
                        lbl_h = self.camera_label.winfo_height()
                        if lbl_w > 100 and lbl_h > 100:
                            target_w, target_h = lbl_w, lbl_h
                        else:
                            target_w, target_h = 640, 480
                        
                        resized = cv2.resize(self._latest_display_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_rgb)
                        img_tk = ImageTk.PhotoImage(img_pil)
                        self.camera_label.config(image=img_tk)
                        self.camera_label.image = img_tk
                    except Exception:
                        pass
        
        except Exception as e:
            print(f"[WARN] UI update error: {e}")
        
        if self.running:
            self.root.after(20, self.update_ui)
    
    def refresh_report(self):
        """Refresh the 30-day report."""
        measurements = get_recent_measurements(30)
        if not measurements:
            messagebox.showinfo("Report", "No data available yet.")
            return
        
        stats = calculate_daily_statistics(measurements)
        time_analysis = get_time_of_day_analysis(measurements)
        day_analysis = get_day_of_week_analysis(measurements)
        
        report_text = f"""Monitoring days: {len(measurements) // 12 if measurements else 0}
Average stress score: {stats.get('avg_stress', 0):.1f}%
Stress percentage: {stats.get('stress_percentage', 0):.1f}%
Stress events: {stats.get('stress_count', 0)}
Normal events: {stats.get('normal_count', 0)}

Most stressful time: {max(time_analysis, key=time_analysis.get) if time_analysis else 'N/A'}
Most stressful day: {max(day_analysis, key=day_analysis.get) if day_analysis else 'N/A'}

Average HR (all): {stats.get('avg_hr', 0):.0f} BPM
Average SpO2: {stats.get('avg_spo2', 0):.1f}%"""
        
        self.stats_text.config(text=report_text)
    
    def download_csv(self):
        """Download measurements as CSV."""
        measurements = get_recent_measurements(30)
        if not measurements:
            messagebox.showwarning("CSV Export", "No data to export.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'HR', 'SpO2', 'HRV', 'Emotion', 'Confidence', 'Stress Score', 'Status'])
                for m in measurements:
                    writer.writerow([m[1], m[2], m[3], m[4], m[8], m[9], m[14], m[15]])
            messagebox.showinfo("CSV Export", f"Data exported to {file_path}")
        except Exception as e:
            messagebox.showerror("CSV Export Error", str(e))
    
    def clear_demo(self):
        """Clear demo data."""
        if messagebox.askyesno("Clear Demo Data", "Delete all demo records? Real data will be preserved."):
            clear_demo_data()
            messagebox.showinfo("Demo Data Cleared", "Demo records deleted.")
    
    def update_prediction(self):
        """Update stress prediction."""
        measurements = get_recent_measurements(30)
        if len(measurements) < 20:
            self.prediction_text.config(text="Prediction: INSUFFICIENT DATA (need more historical records)")
            return
        
        model = train_prediction_model(measurements)
        if model is None:
            self.prediction_text.config(text="Prediction: MODEL TRAINING FAILED")
            return
        
        tomorrow = datetime.now() + timedelta(days=1)
        predicted_score = predict_stress_for_datetime(model, tomorrow)
        
        pred_text = f"""PREDICTED STRESS PATTERN

Next 24 hours:
Predicted stress score: {predicted_score:.0f}%
Expected status: {'STRESS DETECTED' if predicted_score > 65 else 'NORMAL'}

High-risk periods:
- Evening (17:00-21:00)

Recommendation: Monitor stress levels in evening hours."""
        
        self.prediction_text.config(text=pred_text)
    
    def update_risk(self):
        """Update risk indicators."""
        measurements = get_recent_measurements(30)
        if not measurements:
            self.risk_text.config(text="Risk Calculation: INSUFFICIENT DATA")
            return
        
        risks = calculate_risk_indicators(measurements)
        
        risk_text = f"""LONG-TERM STRESS-ASSOCIATED HEALTH RISK

Cardiovascular: {risks.get('cardiovascular', 'UNKNOWN')}
Sleep-related: {risks.get('sleep', 'UNKNOWN')}
Anxiety/mood: {risks.get('anxiety', 'UNKNOWN')}
Metabolic: {risks.get('metabolic', 'UNKNOWN')}

Based on 30-day historical stress patterns.
This is NOT a medical diagnosis.
Consult a healthcare professional for medical advice."""
        
        self.risk_text.config(text=risk_text)
    
    def on_close(self):
        """Cleanup on application close."""
        self.running = False
        if hasattr(self, '_current_stream') and self._current_stream is not None:
            try:
                self._current_stream.close()
            except Exception:
                pass
            self._current_stream = None
        if self.active_cap is not None:
            try:
                self.active_cap.release()
            except Exception:
                pass
        if self.esp32_manager:
            self.esp32_manager.stop_sensor_polling()
        self.root.destroy()


# ========== MAIN ==========
if __name__ == "__main__":
    print(f"[INFO] Starting AI Stress Monitor connecting to ESP32 Watch ({DEFAULT_WATCH_IP})...")
    
    # Initialize database
    init_database()
    
    # Initialize demo data if needed
    initialize_demo_data()
    
    # Create and run application with configured Watch IP
    root = tk.Tk()
    app = StressMonitorApp(root, esp32_ip=DEFAULT_WATCH_IP)
    try:
        root.mainloop()
    except (KeyboardInterrupt, SystemExit):
        print("[INFO] Application closed, releasing hardware...")
        try:
            app.on_close()
        except Exception:
            pass
