import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Dict, Tuple, Optional
from .rendering import HealthEnvRenderer

class HeartHealthEnvironment(gym.Env):
    """
    Intelligent Lifestyle Intervention Agent for Heart Disease Prevention
    Simulates a virtual patient with health metrics and recommends interventions
    """
    
    def __init__(self, render_mode: Optional[str] = None):
        super(HeartHealthEnvironment, self).__init__()
        
        # Define action space: 5 types of interventions with discrete levels
        # [exercise, diet, medication, sleep, stress_reduction]
        self.action_space = spaces.MultiDiscrete([3, 3, 2, 3, 3])
        
        # Define observation space: 7 health metrics + 1 risk score
        # [systolic_bp, diastolic_bp, cholesterol, weight, age, smoking_status, stress_level, risk_score]
        self.observation_space = spaces.Box(
            low=np.array([90, 60, 150, 50, 30, 0, 0, 0]),
            high=np.array([180, 120, 300, 150, 80, 1, 10, 100]),
            dtype=np.float32
        )
        
        # Health parameters
        self.health_state = None
        self.max_steps = 365  # One year of daily interventions
        self.current_step = 0
        self.baseline_risk = 0
        
        # Rendering
        self.render_mode = render_mode
        self.renderer = None
        if render_mode == "human":
            self.renderer = HealthEnvRenderer()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial patient state
        """
        super().reset(seed=seed)
        np.random.seed(seed)
        
        # Initialize patient with realistic health metrics
        self.health_state = {
            'systolic_bp': np.random.uniform(120, 160),  # mmHg
            'diastolic_bp': np.random.uniform(80, 100),  # mmHg  
            'cholesterol': np.random.uniform(180, 280),  # mg/dL
            'weight': np.random.uniform(70, 120),        # kg
            'age': np.random.randint(40, 65),           # years
            'smoking_status': np.random.choice([0, 1]), # 0: non-smoker, 1: smoker
            'stress_level': np.random.uniform(3, 8),    # 0-10 scale
            'risk_score': 0
        }
        
        # Calculate initial Framingham risk score
        self.health_state['risk_score'] = self._calculate_framingham_risk()  # FIXED: removed extra 'm'
        self.baseline_risk = self.health_state['risk_score']
        self.current_step = 0
        
        if self.render_mode == "human":
            self.renderer.reset()
            
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step (one day) in the environment
        """
        self.current_step += 1
        
        # Apply interventions and update health state
        self._apply_interventions(action)
        
        # Calculate natural health progression
        self._update_health_dynamics()
        
        # Calculate reward based on risk reduction
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # Update risk score
        self.health_state['risk_score'] = self._calculate_framingham_risk()  # FIXED: removed extra 'm'
        
        info = {
            'current_risk': self.health_state['risk_score'],
            'risk_reduction': self.baseline_risk - self.health_state['risk_score'],
            'step': self.current_step
        }
        
        if self.render_mode == "human":
            self.renderer.render(self.health_state, action, reward)
            
        return self._get_observation(), reward, terminated, truncated, info
    
    def _apply_interventions(self, action: np.ndarray):
        """
        Apply lifestyle interventions to health metrics
        action: [exercise, diet, medication, sleep, stress_reduction]
        """
        exercise, diet, medication, sleep, stress_reduction = action
        
        # Exercise effects (0: sedentary, 1: moderate, 2: vigorous)
        if exercise == 1:  # Moderate exercise
            self.health_state['systolic_bp'] -= 0.1
            self.health_state['weight'] -= 0.05
            self.health_state['stress_level'] -= 0.1
        elif exercise == 2:  # Vigorous exercise
            self.health_state['systolic_bp'] -= 0.2
            self.health_state['weight'] -= 0.1
            self.health_state['stress_level'] -= 0.2
        
        # Dietary effects (0: poor, 1: balanced, 2: optimal)
        if diet == 1:  # Balanced diet
            self.health_state['cholesterol'] -= 0.2
            self.health_state['weight'] -= 0.03
        elif diet == 2:  # Optimal diet
            self.health_state['cholesterol'] -= 0.5
            self.health_state['weight'] -= 0.07
        
        # Medication adherence (0: no, 1: yes)
        if medication == 1:
            self.health_state['systolic_bp'] -= 0.3
            self.health_state['diastolic_bp'] -= 0.2
            self.health_state['cholesterol'] -= 0.8
        
        # Sleep effects (0: poor, 1: adequate, 2: optimal)
        if sleep == 1:  # Adequate sleep
            self.health_state['stress_level'] -= 0.2
            self.health_state['systolic_bp'] -= 0.05
        elif sleep == 2:  # Optimal sleep
            self.health_state['stress_level'] -= 0.4
            self.health_state['systolic_bp'] -= 0.1
        
        # Stress reduction effects (0: none, 1: moderate, 2: high)
        if stress_reduction == 1:
            self.health_state['stress_level'] -= 0.3
            self.health_state['systolic_bp'] -= 0.08
        elif stress_reduction == 2:
            self.health_state['stress_level'] -= 0.6
            self.health_state['systolic_bp'] -= 0.15
        
        # Apply bounds to ensure realistic values
        self._enforce_health_bounds()
    
    def _update_health_dynamics(self):
        """
        Simulate natural health progression and aging effects
        """
        # Natural BP increase with age/stress
        self.health_state['systolic_bp'] += 0.01
        self.health_state['diastolic_bp'] += 0.005
        
        # Cholesterol natural progression
        self.health_state['cholesterol'] += 0.05
        
        # Weight natural progression (slight gain)
        self.health_state['weight'] += 0.01
        
        # Stress natural fluctuations
        self.health_state['stress_level'] += np.random.uniform(-0.1, 0.1)
        
        # Smoking effects if smoker
        if self.health_state['smoking_status'] == 1:
            self.health_state['systolic_bp'] += 0.05
            self.health_state['cholesterol'] += 0.1
        
        # Age progression (slight effect over time)
        self.health_state['age'] += 1/365  # Daily aging
        
        self._enforce_health_bounds()
    
    def _calculate_framingham_risk(self) -> float:  # FIXED: removed extra 'm'
        """
        Calculate simplified Framingham Risk Score for 10-year CVD risk
        Simplified version for demonstration purposes
        """
        risk = 0
        
        # Age factor
        age = self.health_state['age']
        if age >= 60:
            risk += 8
        elif age >= 50:
            risk += 5
        elif age >= 40:
            risk += 3
        
        # Blood pressure factor
        systolic_bp = self.health_state['systolic_bp']
        if systolic_bp >= 160:
            risk += 4
        elif systolic_bp >= 140:
            risk += 3
        elif systolic_bp >= 130:
            risk += 2
        elif systolic_bp >= 120:
            risk += 1
        
        # Cholesterol factor
        cholesterol = self.health_state['cholesterol']
        if cholesterol >= 240:
            risk += 4
        elif cholesterol >= 200:
            risk += 3
        elif cholesterol >= 180:
            risk += 2
        
        # Smoking factor
        if self.health_state['smoking_status'] == 1:
            risk += 4
        
        # Weight/BMI factor (simplified)
        bmi = self.health_state['weight'] / ((1.7) ** 2)  # Approximate BMI
        if bmi >= 30:
            risk += 3
        elif bmi >= 25:
            risk += 2
        
        # Stress factor
        if self.health_state['stress_level'] >= 7:
            risk += 2
        elif self.health_state['stress_level'] >= 5:
            risk += 1
        
        return min(risk * 1.5, 100)  # Scale to 0-100 range
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on risk reduction and healthy behaviors
        """
        current_risk = self.health_state['risk_score']
        risk_reduction = self.baseline_risk - current_risk
        
        # Primary reward: risk reduction
        reward = risk_reduction * 10
        
        # Bonus for maintaining healthy metrics
        if self.health_state['systolic_bp'] < 130:
            reward += 1
        if self.health_state['cholesterol'] < 200:
            reward += 1
        if self.health_state['weight'] < 90:  # Reasonable weight threshold
            reward += 1
        if self.health_state['stress_level'] < 5:
            reward += 1
        
        # Penalty for extremely unhealthy states
        if self.health_state['systolic_bp'] > 160:
            reward -= 2
        if self.health_state['cholesterol'] > 240:
            reward -= 2
        if self.health_state['stress_level'] > 8:
            reward -= 1
        
        return reward
    
    def _is_terminated(self) -> bool:
        """
        Check if episode should terminate (critical health condition)
        """
        # Terminate if critical health conditions
        if (self.health_state['systolic_bp'] > 180 or 
            self.health_state['diastolic_bp'] > 120 or
            self.health_state['risk_score'] > 80):
            return True
        
        return False
    
    def _enforce_health_bounds(self):
        """
        Ensure health metrics stay within realistic bounds
        """
        self.health_state['systolic_bp'] = np.clip(self.health_state['systolic_bp'], 90, 180)
        self.health_state['diastolic_bp'] = np.clip(self.health_state['diastolic_bp'], 60, 120)
        self.health_state['cholesterol'] = np.clip(self.health_state['cholesterol'], 150, 300)
        self.health_state['weight'] = np.clip(self.health_state['weight'], 50, 150)
        self.health_state['stress_level'] = np.clip(self.health_state['stress_level'], 0, 10)
    
    def _get_observation(self) -> np.ndarray:
        """
        Convert health state to observation array
        """
        return np.array([
            self.health_state['systolic_bp'],
            self.health_state['diastolic_bp'], 
            self.health_state['cholesterol'],
            self.health_state['weight'],
            self.health_state['age'],
            self.health_state['smoking_status'],
            self.health_state['stress_level'],
            self.health_state['risk_score']
        ], dtype=np.float32)
    
    def render(self):
        """Render environment"""
        if self.render_mode == "human" and self.renderer:
            self.renderer.render(self.health_state, None, 0)
    
    def close(self):
        """Clean up resources"""
        if self.renderer:
            self.renderer.close()