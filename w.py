import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import os
import csv
import math

def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure value is in valid range
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_distance(a, b):
    """Calculate distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))

class ExerciseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.exercise_type = "none"
        self.rep_count = 0
        self.position = None
        self.feedback = ""
        self.last_feedback_time = 0
        self.feedback_duration = 3
        self.exercise_start_time = None
        self.last_rep_time = None
        self.rep_times = []
        self.form_score = 100  # Starting with perfect score
        
        # Session stats
        self.session_start = time.time()
        self.session_stats = {}
        self.workout_history = []
        
        # UI elements
        self.show_angles = False
        self.show_skeleton = True
        self.tutorial_mode = False
        self.tutorial_step = 0
        self.calories_burned = 0
        
        # Auto detection variables
        self.movement_history = []
        self.detection_window = 30  # frames to analyze for exercise detection
        self.auto_detect = False
        self.detection_confidence = 0
        
        # Create directory for saving workout data
        os.makedirs("workout_data", exist_ok=True)
        
        # Initialize the complete exercise catalog
        self.init_exercise_catalog()
    
    def init_exercise_catalog(self):
        """Initialize the catalog of supported exercises with form criteria."""
        self.exercise_catalog = {
            "pushup": {
                "name": "Push-ups",
                "target_muscles": ["chest", "triceps", "shoulders"],
                "calories_per_rep": 0.5,
                "thresholds": {
                    "elbow_down": 90,
                    "elbow_up": 160,
                    "back_min": 150,
                    "hip_min": 160
                },
                "detection_pattern": {
                    "primary_angle": "elbow",
                    "range": [90, 160],
                    "rep_direction": "up-down-up"
                },
                "instructions": [
                    "Start in a plank position with arms straight",
                    "Lower your body until elbows reach 90 degrees",
                    "Keep your back straight throughout the movement",
                    "Push back up to the starting position"
                ]
            },
            "squat": {
                "name": "Squats",
                "target_muscles": ["quadriceps", "hamstrings", "glutes"],
                "calories_per_rep": 0.4,
                "thresholds": {
                    "knee_down": 110,
                    "knee_up": 170,
                    "hip_down": 100,
                    "hip_up": 170,
                    "back_min": 145
                },
                "detection_pattern": {
                    "primary_angle": "knee",
                    "range": [90, 170],
                    "rep_direction": "up-down-up"
                },
                "instructions": [
                    "Stand with feet shoulder-width apart",
                    "Bend knees and lower to 90 degrees",
                    "Keep back straight and chest up",
                    "Return to standing position"
                ]
            },
            "plank": {
                "name": "Plank",
                "target_muscles": ["core", "shoulders", "back"],
                "calories_per_min": 3.5,
                "thresholds": {
                    "elbow_ideal": 90,
                    "back_min": 160,
                    "shoulder_ideal": 80
                },
                "detection_pattern": {
                    "primary_angle": "back",
                    "range": [160, 180],
                    "position_hold": True
                },
                "instructions": [
                    "Rest on forearms with elbows under shoulders",
                    "Form a straight line from head to heels",
                    "Engage core and maintain neutral spine",
                    "Hold the position as long as possible"
                ]
            },
            "lunge": {
                "name": "Lunges",
                "target_muscles": ["quadriceps", "hamstrings", "glutes", "calves"],
                "calories_per_rep": 0.3,
                "thresholds": {
                    "front_knee": 90,
                    "back_knee": 100,
                    "torso_min": 150
                },
                "detection_pattern": {
                    "primary_angle": "knee",
                    "range": [90, 170],
                    "rep_direction": "up-down-up",
                    "asymmetric": True
                },
                "instructions": [
                    "Stand upright with feet hip-width apart",
                    "Step forward with one leg",
                    "Lower your body until both knees form 90-degree angles",
                    "Push back up and return to start position"
                ]
            },
            "bicep_curl": {
                "name": "Bicep Curls",
                "target_muscles": ["biceps", "forearms"],
                "calories_per_rep": 0.2,
                "thresholds": {
                    "elbow_down": 160,
                    "elbow_up": 50,
                    "shoulder_stable": 20  # Max movement of shoulder
                },
                "detection_pattern": {
                    "primary_angle": "elbow",
                    "range": [40, 160],
                    "rep_direction": "down-up-down",
                    "can_be_asymmetric": True
                },
                "instructions": [
                    "Stand with weights at your sides, palms forward",
                    "Keep elbows close to your torso",
                    "Curl the weights up to shoulder level",
                    "Lower the weights back to the starting position"
                ]
            },
            "shoulder_press": {
                "name": "Shoulder Press",
                "target_muscles": ["shoulders", "triceps"],
                "calories_per_rep": 0.3,
                "thresholds": {
                    "elbow_down": 90,
                    "elbow_up": 170,
                    "shoulder_min": 150
                },
                "detection_pattern": {
                    "primary_angle": "elbow",
                    "range": [90, 170],
                    "rep_direction": "down-up-down",
                    "position": "arms_up"
                },
                "instructions": [
                    "Start with elbows bent at 90 degrees at shoulder height",
                    "Press weights up until arms are fully extended",
                    "Keep core engaged and avoid arching your back",
                    "Lower weights back to shoulder height"
                ]
            },
            "jumping_jack": {
                "name": "Jumping Jacks",
                "target_muscles": ["shoulders", "calves", "cardio"],
                "calories_per_rep": 0.2,
                "thresholds": {
                    "arm_range": [20, 150],
                    "leg_min_width": 0.3  # Relative to height
                },
                "detection_pattern": {
                    "primary_pattern": "arm_spread",
                    "secondary_pattern": "leg_spread",
                    "rep_direction": "in-out-in"
                },
                "instructions": [
                    "Stand upright with feet together and arms at sides",
                    "Jump while spreading legs and raising arms above head",
                    "Jump again to return to starting position",
                    "Maintain a steady rhythm"
                ]
            },
            "sit_up": {
                "name": "Sit-ups",
                "target_muscles": ["abs", "hip flexors"],
                "calories_per_rep": 0.25,
                "thresholds": {
                    "hip_min": 90,
                    "hip_max": 150,
                    "shoulder_hip_approach": 50  # Min distance between shoulder and hip
                },
                "detection_pattern": {
                    "primary_angle": "hip",
                    "range": [90, 150],
                    "rep_direction": "up-down-up",
                    "position": "lying"
                },
                "instructions": [
                    "Lie on your back with knees bent and feet flat",
                    "Place hands behind your head or across chest",
                    "Curl your upper body toward your knees",
                    "Lower back down with control"
                ]
            },
            "lateral_raise": {
                "name": "Lateral Raises",
                "target_muscles": ["lateral deltoids", "shoulders"],
                "calories_per_rep": 0.15,
                "thresholds": {
                    "arm_min": 20,
                    "arm_max": 90,
                    "elbow_min": 160
                },
                "detection_pattern": {
                    "primary_angle": "shoulder",
                    "range": [20, 90],
                    "rep_direction": "down-up-down"
                },
                "instructions": [
                    "Stand with feet shoulder-width apart",
                    "Hold weights at your sides",
                    "Raise arms out to sides until parallel to floor",
                    "Lower arms slowly back to starting position"
                ]
            },
            "tricep_extension": {
                "name": "Tricep Extensions",
                "target_muscles": ["triceps"],
                "calories_per_rep": 0.15,
                "thresholds": {
                    "elbow_min": 80,
                    "elbow_max": 160,
                    "upper_arm_stable": 30
                },
                "detection_pattern": {
                    "primary_angle": "elbow",
                    "range": [80, 160],
                    "rep_direction": "up-down-up",
                    "position": "arms_up"
                },
                "instructions": [
                    "Hold weight with both hands behind your head",
                    "Keep upper arms stable pointing to ceiling",
                    "Extend arms fully by straightening elbows",
                    "Lower weight back behind head with control"
                ]
            }
        }
    
    def get_landmarks(self, landmarks):
        """Extract key landmarks from pose detection."""
        points = {}
        
        if landmarks:
            # Left side points
            points["l_shoulder"] = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            points["l_elbow"] = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            points["l_wrist"] = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                               landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            points["l_hip"] = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            points["l_knee"] = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            points["l_ankle"] = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Right side points
            points["r_shoulder"] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            points["r_elbow"] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            points["r_wrist"] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            points["r_hip"] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            points["r_knee"] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            points["r_ankle"] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                               
            # Additional points
            points["nose"] = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                            landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
            points["l_ear"] = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]
            points["r_ear"] = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                            
        return points
    
    def calculate_angles(self, points):
        """Calculate relevant angles for exercise form analysis."""
        angles = {}
        
        # Left side angles
        angles["l_elbow"] = calculate_angle(points["l_shoulder"], points["l_elbow"], points["l_wrist"])
        angles["l_shoulder"] = calculate_angle(points["l_hip"], points["l_shoulder"], points["l_elbow"])
        angles["l_hip"] = calculate_angle(points["l_shoulder"], points["l_hip"], points["l_knee"])
        angles["l_knee"] = calculate_angle(points["l_hip"], points["l_knee"], points["l_ankle"])
        
        # Right side angles
        angles["r_elbow"] = calculate_angle(points["r_shoulder"], points["r_elbow"], points["r_wrist"])
        angles["r_shoulder"] = calculate_angle(points["r_hip"], points["r_shoulder"], points["r_elbow"])
        angles["r_hip"] = calculate_angle(points["r_shoulder"], points["r_hip"], points["r_knee"])
        angles["r_knee"] = calculate_angle(points["r_hip"], points["r_knee"], points["r_ankle"])
        
        # Average angles for general use
        angles["elbow"] = (angles["l_elbow"] + angles["r_elbow"]) / 2
        angles["shoulder"] = (angles["l_shoulder"] + angles["r_shoulder"]) / 2
        angles["hip"] = (angles["l_hip"] + angles["r_hip"]) / 2
        angles["knee"] = (angles["l_knee"] + angles["r_knee"]) / 2
        
        # Back angle calculation
        shoulder_mid_x = (points["l_shoulder"][0] + points["r_shoulder"][0]) / 2
        shoulder_mid_y = (points["l_shoulder"][1] + points["r_shoulder"][1]) / 2
        hip_mid_x = (points["l_hip"][0] + points["r_hip"][0]) / 2
        hip_mid_y = (points["l_hip"][1] + points["r_hip"][1]) / 2
        
        # Calculate vertical line from hips
        vertical_point = [hip_mid_x, hip_mid_y - 0.5]
        
        # Back angle relative to vertical
        angles["back"] = calculate_angle(vertical_point, [hip_mid_x, hip_mid_y], [shoulder_mid_x, shoulder_mid_y])
        
        # Head angle (for neck position)
        ear_mid_x = (points["l_ear"][0] + points["r_ear"][0]) / 2
        ear_mid_y = (points["l_ear"][1] + points["r_ear"][1]) / 2
        angles["neck"] = calculate_angle([shoulder_mid_x, shoulder_mid_y - 0.5], 
                                         [shoulder_mid_x, shoulder_mid_y], 
                                         [ear_mid_x, ear_mid_y])
        
        # Calculate leg spread for jumping jacks
        leg_width = abs(points["l_ankle"][0] - points["r_ankle"][0])
        shoulder_width = abs(points["l_shoulder"][0] - points["r_shoulder"][0])
        angles["leg_spread_ratio"] = leg_width / shoulder_width if shoulder_width > 0 else 0
        
        # Arms spread angle for jumping jacks
        angles["arm_spread"] = calculate_angle(points["l_wrist"], 
                                             [shoulder_mid_x, shoulder_mid_y], 
                                             points["r_wrist"])
        
        return angles
    
    def detect_exercise(self, angles_history):
        """Auto-detect which exercise is being performed based on movement patterns."""
        if len(angles_history) < self.detection_window:
            return None, 0
        
        # Extract key angle patterns from recent history
        recent_angles = angles_history[-self.detection_window:]
        
        # Calculate ranges and patterns for key angles
        angle_ranges = {}
        for key in ["elbow", "knee", "hip", "shoulder", "back", "arm_spread"]:
            values = [frame.get(key, 0) for frame in recent_angles if key in frame]
            if values:
                angle_ranges[key] = {
                    "min": min(values),
                    "max": max(values),
                    "range": max(values) - min(values),
                    "std": np.std(values) if len(values) > 1 else 0
                }
        
        # Check for significant movement in key angles
        significant_movement = {k: v["range"] > 30 for k, v in angle_ranges.items() if k in angle_ranges}
        
        # Score each exercise based on movement patterns
        exercise_scores = {}
        
        for ex_name, ex_data in self.exercise_catalog.items():
            score = 0
            pattern = ex_data.get("detection_pattern", {})
            
            # Check if primary angle shows significant movement
            primary_angle = pattern.get("primary_angle", "")
            if primary_angle in angle_ranges:
                if pattern.get("position_hold", False):
                    # For static exercises like planks
                    if angle_ranges[primary_angle]["std"] < 5:  # Low variation
                        score += 40
                        
                        # Check if angle is in expected range
                        expected_range = pattern.get("range", [0, 180])
                        angle_avg = (angle_ranges[primary_angle]["min"] + angle_ranges[primary_angle]["max"]) / 2
                        if expected_range[0] <= angle_avg <= expected_range[1]:
                            score += 40
                else:
                    # For dynamic exercises
                    if significant_movement.get(primary_angle, False):
                        score += 30
                        
                        # Check if movement range matches expected
                        expected_range = pattern.get("range", [0, 180])
                        if (expected_range[0] <= angle_ranges[primary_angle]["min"] <= expected_range[1] and
                            expected_range[0] <= angle_ranges[primary_angle]["max"] <= expected_range[1]):
                            score += 40
                            
                            # Additional score if range is significant
                            expected_movement = expected_range[1] - expected_range[0]
                            if angle_ranges[primary_angle]["range"] > expected_movement * 0.7:
                                score += 20
            
            # Check secondary patterns
            if ex_name == "jumping_jack" and "arm_spread" in angle_ranges and "leg_spread_ratio" in angle_ranges:
                if angle_ranges["arm_spread"]["range"] > 40 and significant_movement.get("arm_spread", False):
                    score += 40
                    if angle_ranges.get("leg_spread_ratio", {}).get("max", 0) > 1:
                        score += 40
            
            exercise_scores[ex_name] = score
        
        # Get best match
        best_exercise = max(exercise_scores.items(), key=lambda x: x[1])
        
        # Only return if confidence is high enough (score > 60)
        if best_exercise[1] > 60:
            return best_exercise[0], best_exercise[1]/100
        else:
            return None, 0
    
    def track_generic_exercise(self, angles, exercise_type):
        """Track form and count reps for any exercise based on its catalog definition."""
        ex_data = self.exercise_catalog.get(exercise_type, {})
        thresholds = ex_data.get("thresholds", {})
        pattern = ex_data.get("detection_pattern", {})
        feedback = ""
        
        # Handle different exercise types based on detection pattern
        primary_angle = pattern.get("primary_angle", "")
        
        # For static exercises like planks
        if pattern.get("position_hold", False):
            # Check posture for static exercises
            if primary_angle == "back" and "back_min" in thresholds:
                if angles["back"] < thresholds["back_min"]:
                    feedback = "Straighten your back!"
                    self.form_score -= 0.5
            
            # For plank specific checks
            if exercise_type == "plank":
                elbow_angle = angles["elbow"]
                shoulder_angle = angles["shoulder"]
                
                if abs(elbow_angle - thresholds["elbow_ideal"]) > 15:
                    feedback = "Adjust elbow position to 90 degrees!"
                    self.form_score -= 0.5
                elif abs(shoulder_angle - thresholds["shoulder_ideal"]) > 15:
                    feedback = "Adjust shoulder position!"
                    self.form_score -= 0.5
                
            # Calculate duration instead of reps for hold exercises
            if self.exercise_start_time is None:
                self.exercise_start_time = time.time()
                
            # Update calories for time-based exercises
            elapsed = time.time() - self.exercise_start_time
            calories_per_min = ex_data.get("calories_per_min", 3.0)
            self.calories_burned = (elapsed / 60) * calories_per_min
            
            return feedback
        
        # For dynamic exercises (most exercises)
        else:
            # Get angle ranges for the exercise
            angle_min, angle_max = pattern.get("range", [0, 180])
            
            # Check form based on exercise type
            if exercise_type == "pushup":
                if angles["back"] < thresholds["back_min"]:
                    feedback = "Keep your back straight!"
                    self.form_score -= 1
            
            elif exercise_type == "squat":
                if angles["back"] < thresholds["back_min"]:
                    feedback = "Keep your back straighter!"
                    self.form_score -= 1
                if angles["knee"] < 80:  # Too deep
                    feedback = "Don't go too deep, protect your knees!"
                    self.form_score -= 1
            
            elif exercise_type == "lunge":
                if angles["back"] < thresholds["torso_min"]:
                    feedback = "Keep your torso upright!"
                    self.form_score -= 1
            
            elif exercise_type == "bicep_curl":
                # Check if shoulders are moving too much
                shoulder_movement = abs(angles["l_shoulder"] - angles["r_shoulder"])
                if shoulder_movement > thresholds["shoulder_stable"]:
                    feedback = "Keep your shoulders still!"
                    self.form_score -= 1
            
            elif exercise_type == "shoulder_press":
                if angles["back"] < thresholds["shoulder_min"]:
                    feedback = "Watch your back arch!"
                    self.form_score -= 1
            
            # Get primary angle value for rep counting
            angle_value = angles.get(primary_angle, 0)
            
            # Determine rep direction (different exercises count reps at different points)
            rep_direction = pattern.get("rep_direction", "up-down-up")
            
            # Count reps based on pattern
            if rep_direction == "up-down-up":
                if angle_value < angle_min + (angle_max - angle_min) * 0.3 and self.position == "up":
                    self.position = "down"
                elif angle_value > angle_max - (angle_max - angle_min) * 0.3 and self.position == "down":
                    self.rep_count += 1
                    self.position = "up"
                    
                    # Record rep time for consistency tracking
                    current_time = time.time()
                    if self.last_rep_time:
                        self.rep_times.append(current_time - self.last_rep_time)
                    self.last_rep_time = current_time
                    
                    # Update calories burned
                    calories_per_rep = ex_data.get("calories_per_rep", 0.3)
                    self.calories_burned += calories_per_rep
            
            elif rep_direction == "down-up-down":
                if angle_value > angle_max - (angle_max - angle_min) * 0.3 and self.position == "down":
                    self.position = "up"
                elif angle_value < angle_min + (angle_max - angle_min) * 0.3 and self.position == "up":
                    self.rep_count += 1
                    self.position = "down"
                    
                    # Record rep time
                    current_time = time.time()
                    if self.last_rep_time:
                        self.rep_times.append(current_time - self.last_rep_time)
                    self.last_rep_time = current_time
                    
                    # Update calories burned
                    calories_per_rep = ex_data.get("calories_per_rep", 0.3)
                    self.calories_burned += calories_per_rep
            
            # Handle jumping jacks specifically
            elif exercise_type == "jumping_jack":
                arm_spread = angles["arm_spread"]
                leg_spread = angles["leg_spread_ratio"]
                
                if arm_spread > 120 and leg_spread > 1.0 and self.position == "in":
                    self.position = "out"
                elif arm_spread < 60 and leg_spread < 0.5 and self.position == "out":
                    self.rep_count += 1
                    self.position = "in"
                    
                    # Update calories
                    calories_per_rep = ex_data.get("calories_per_rep", 0.2)
                    self.calories_burned += calories_per_rep
            
            # Initialize position if None
            if self.position is None:
                if rep_direction == "up-down-up":
                    if angle_value > angle_max - (angle_max - angle_min) * 0.3:
                        self.position = "up"
                    else:
                        self.position = "down"
                elif rep_direction == "down-up-down":
                    if angle_value < angle_min + (angle_max - angle_min) * 0.3:
                        self.position = "down"
                    else:
                        self.position = "up"
                elif exercise_type == "jumping_jack":
                    if angles["arm_spread"] < 60:
                        self.position = "in"
                    else:
                        self.position = "out"
            
            return feedback
    
    def draw_tutorial(self, image, exercise_type):
        """Draw tutorial instructions on the image."""
        ex_data = self.exercise_catalog.get(exercise_type, {})
        instructions = ex_data.get("instructions", ["No instructions available"])
        
        # Background for tutorial
        cv2.rectangle(image, (10, image.shape[0] - 160), (image.shape[1] - 10, image.shape[0] - 10), (0, 0, 0), -1)
        cv2.rectangle(image, (10, image.shape[0] - 160), (image.shape[1] - 10, image.shape[0] - 10), (255, 255, 255), 2)
        
        # Title
        cv2.putText(image, f"HOW TO: {ex_data.get('name', exercise_type.upper())}", 
                    (30, image.shape[0] - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        for i, instr in enumerate(instructions):
            cv2.putText(image, f"{i+1}. {instr}", 
                        (30, image.shape[0] - 100 + (i * 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Press any key prompt
        cv2.putText(image, "Press 'T' to close tutorial", 
                    (image.shape[1] - 250, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def draw_interface(self, image, angles):
        """Draw UI elements on the image."""
        h, w = image.shape[:2]
        
        # Draw panel backgrounds
        # Top info panel
        cv2.rectangle(image, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.rectangle(image, (0, 0), (w, 50), (255, 255, 255), 1)
        
# Right side panel
        right_panel_width = 250
        cv2.rectangle(image, (w - right_panel_width, 50), (w, h - 50), (0, 0, 0), -1)
        cv2.rectangle(image, (w - right_panel_width, 50), (w, h - 50), (255, 255, 255), 1)
        
        # Bottom control panel
        cv2.rectangle(image, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.rectangle(image, (0, h - 50), (w, h), (255, 255, 255), 1)
        
        # Top title bar
        ex_name = self.exercise_catalog.get(self.exercise_type, {}).get("name", self.exercise_type.upper())
        cv2.putText(image, f"EXERCISE: {ex_name}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.auto_detect:
            status = "AUTO-DETECT" if self.exercise_type == "none" else f"DETECTED: {ex_name} ({int(self.detection_confidence*100)}%)"
            cv2.putText(image, status, (w - 380, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Stats in right panel
        y_pos = 80
        cv2.putText(image, "STATS", (w - right_panel_width + 20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 30
        
        # Rep count
        if self.exercise_type != "plank":
            cv2.putText(image, f"Reps: {self.rep_count}", (w - right_panel_width + 20, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 30
        else:
            # Duration for plank
            duration = time.time() - self.exercise_start_time if self.exercise_start_time else 0
            minutes, seconds = divmod(int(duration), 60)
            cv2.putText(image, f"Duration: {minutes:02d}:{seconds:02d}", (w - right_panel_width + 20, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 30
        
        # Position
        if self.position and self.exercise_type != "plank":
            cv2.putText(image, f"Position: {self.position}", (w - right_panel_width + 20, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 30
        
        # Form score
        form_color = (0, 255, 0) if self.form_score > 80 else (0, 255, 255) if self.form_score > 60 else (0, 0, 255)
        cv2.putText(image, f"Form: {int(self.form_score)}%", (w - right_panel_width + 20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, form_color, 2)
        y_pos += 30
        
        # Calories
        cv2.putText(image, f"Calories: {self.calories_burned:.1f}", (w - right_panel_width + 20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 40
        
        # Target muscles
        target_muscles = self.exercise_catalog.get(self.exercise_type, {}).get("target_muscles", [])
        if target_muscles:
            cv2.putText(image, "Target Muscles:", (w - right_panel_width + 20, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 25
            for muscle in target_muscles:
                cv2.putText(image, f"- {muscle.capitalize()}", (w - right_panel_width + 30, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 20
        
        # Feedback
        if self.feedback:
            # Background for feedback
            feedback_y = h - 100
            cv2.rectangle(image, (10, feedback_y - 40), (w - right_panel_width - 10, feedback_y + 10), (0, 0, 100), -1)
            cv2.rectangle(image, (10, feedback_y - 40), (w - right_panel_width - 10, feedback_y + 10), (255, 255, 255), 2)
            
            cv2.putText(image, "FORM FEEDBACK:", (20, feedback_y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, self.feedback, (20, feedback_y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show angles if enabled
        if self.show_angles:
            ang_y = 60
            for key, angle in angles.items():
                if key in ["l_elbow", "r_elbow", "l_knee", "r_knee", "back", "neck"]:  # Show only main angles
                    cv2.putText(image, f'{key}: {int(angle)}°', (10, ang_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    ang_y += 20
        
        # Bottom controls guide
        controls = [
            "Q: Quit", "R: Reset", "A: Auto-detect", 
            "T: Tutorial", "S: Show/Hide Skeleton", "D: Show/Hide Angles"
        ]
        
        ctrl_x = 10
        for ctrl in controls:
            cv2.putText(image, ctrl, (ctrl_x, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            ctrl_x += w // len(controls)
    
    def save_workout_data(self):
        """Save workout statistics to CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"workout_data/workout_{timestamp}.csv"
        
        # Calculate additional stats
        duration = time.time() - self.session_start
        avg_rep_time = sum(self.rep_times) / len(self.rep_times) if self.rep_times else 0
        
        # Compile workout data
        workout_data = {
            "timestamp": timestamp,
            "exercise": self.exercise_catalog.get(self.exercise_type, {}).get("name", self.exercise_type),
            "reps": self.rep_count,
            "duration_seconds": duration,
            "calories_burned": self.calories_burned,
            "form_score": self.form_score,
            "avg_rep_time": avg_rep_time
        }
        
        # Save to CSV
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = workout_data.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerow(workout_data)
        
        print(f"Workout data saved to {filename}")
        return workout_data
    
    def process_frame(self, frame, results):
        """Process pose detection results and track exercise."""
        image = frame.copy()
        
        # Initialize angles data
        angles = {}
        
        if results.pose_landmarks:
            # Get landmarks and calculate angles
            landmarks = results.pose_landmarks.landmark
            points = self.get_landmarks(landmarks)
            angles = self.calculate_angles(points)
            
            # Draw skeleton if enabled
            if self.show_skeleton:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # Auto-detect exercise if enabled
            if self.auto_detect and self.exercise_type == "none":
                # Store angle data for detection
                self.movement_history.append(angles)
                if len(self.movement_history) > self.detection_window:
                    self.movement_history.pop(0)
                
                # Try to detect exercise every 15 frames
                if len(self.movement_history) % 15 == 0:
                    detected_exercise, confidence = self.detect_exercise(self.movement_history)
                    if detected_exercise and confidence > 0.6:
                        self.exercise_type = detected_exercise
                        self.detection_confidence = confidence
                        self.position = None
                        self.rep_count = 0
                        self.form_score = 100
                        self.calories_burned = 0
                        print(f"Detected exercise: {detected_exercise} (Confidence: {confidence:.2f})")
            
            # Track exercise based on selected type
            if self.exercise_type != "none":
                feedback = self.track_generic_exercise(angles, self.exercise_type)
                
                # Update feedback if new
                if feedback and time.time() - self.last_feedback_time > self.feedback_duration:
                    self.feedback = feedback
                    self.last_feedback_time = time.time()
            
            # Clear feedback if duration expired
            if time.time() - self.last_feedback_time > self.feedback_duration:
                self.feedback = ""
        
        # Draw UI elements
        self.draw_interface(image, angles)
        
        # Draw tutorial if enabled
        if self.tutorial_mode and self.exercise_type != "none":
            self.draw_tutorial(image, self.exercise_type)
        
        return image

def main():
    cap = cv2.VideoCapture(0)
    
    # Improve camera settings if available
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    tracker = ExerciseTracker()
    
    # Set up a display window
    cv2.namedWindow('Smart Exercise Trainer', cv2.WINDOW_NORMAL)
    
    print("\n=== SMART EXERCISE TRAINER ===")
    print("Select exercise type or use auto-detection:")
    print("1: Push-ups")
    print("2: Squats")
    print("3: Plank")
    print("4: Lunges")
    print("5: Bicep Curls")
    print("6: Shoulder Press")
    print("7: Jumping Jacks")
    print("8: Sit-ups")
    print("9: Lateral Raises")
    print("0: Tricep Extensions")
    print("A: Auto-detect")
    choice = input("Enter choice: ")
    
    exercise_map = {
        "1": "pushup",
        "2": "squat",
        "3": "plank",
        "4": "lunge",
        "5": "bicep_curl",
        "6": "shoulder_press",
        "7": "jumping_jack",
        "8": "sit_up",
        "9": "lateral_raise",
        "0": "tricep_extension"
    }
    
    if choice.lower() == 'a':
        tracker.auto_detect = True
        print("Auto-detection enabled. Start performing an exercise.")
    elif choice in exercise_map:
        tracker.exercise_type = exercise_map[choice]
        print(f"Selected exercise: {tracker.exercise_catalog[tracker.exercise_type]['name']}")
    else:
        print("Invalid choice. Auto-detection enabled.")
        tracker.auto_detect = True
    
    # Mediapipe setup
    with mp.solutions.pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip the image horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            
            # To improve performance, optionally mark the image as not writeable
            frame.flags.writeable = False
            
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image and get pose landmarks
            results = pose.process(image_rgb)
            
            # Draw the pose annotation on the image
            frame.flags.writeable = True
            
            # Process frame with tracker
            output_image = tracker.process_frame(frame, results)
            
            # Display the resulting frame
            cv2.imshow('Smart Exercise Trainer', output_image)
            
            # Key handling
            key = cv2.waitKey(10) & 0xFF
            
            # Press 'q' to exit
            if key == ord('q'):
                # Save workout data before exiting
                if tracker.exercise_type != "none" and tracker.rep_count > 0:
                    tracker.save_workout_data()
                break
            
            # Press 'r' to reset counter
            elif key == ord('r'):
                tracker.rep_count = 0
                tracker.position = None
                tracker.form_score = 100
                tracker.calories_burned = 0
                tracker.exercise_start_time = None if tracker.exercise_type == "plank" else tracker.exercise_start_time
            
            # Press 't' to toggle tutorial
            elif key == ord('t'):
                tracker.tutorial_mode = not tracker.tutorial_mode
            
            # Press 's' to toggle skeleton
            elif key == ord('s'):
                tracker.show_skeleton = not tracker.show_skeleton
            
            # Press 'd' to toggle angle display
            elif key == ord('d'):
                tracker.show_angles = not tracker.show_angles
            
            # Press 'a' to toggle auto-detection
            elif key == ord('a'):
                tracker.auto_detect = not tracker.auto_detect
                if tracker.auto_detect:
                    # Reset when enabling auto-detection
                    tracker.exercise_type = "none"
                    tracker.movement_history = []
            
            # Number keys to change exercise
            for k, ex_type in exercise_map.items():
                if key == ord(k):
                    tracker.exercise_type = ex_type
                    tracker.rep_count = 0
                    tracker.position = None
                    tracker.form_score = 100
                    tracker.calories_burned = 0
                    tracker.auto_detect = False
                    tracker.exercise_start_time = None if ex_type == "plank" else tracker.exercise_start_time
                    print(f"Changed to: {tracker.exercise_catalog[ex_type]['name']}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
