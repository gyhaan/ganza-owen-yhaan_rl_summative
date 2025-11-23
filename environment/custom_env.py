# environment/custom_env.py  ← FINAL VERSION – PERFECT UI + LAST_ACTION + NO ERRORS
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import math

pygame.init()  # Keep this at the top

class HeartHealthEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(12)

        self.current_step = 0
        self.max_steps = 520
        self.last_action = 0  # ← NEW: tracks current action for display

    def _get_obs(self):
        return np.array([
            self.age_norm, self.bp_norm, self.chol_norm, self.hdl_norm,
            self.bmi_norm, self.smoking, self.stress,
            self.exercise_level, self.diet_quality, self.risk_score
        ], dtype=np.float32)

    def _calculate_risk(self):
        age = self.age
        bp = self.sbp
        chol = self.total_chol
        hdl = self.hdl
        bmi = self.bmi
        smoke = 1 if self.smoking > 0.3 else 0

        risk = 0.05
        risk += 0.02 * (age - 50) / 10
        risk += 0.03 * max(0, bp - 120) / 20
        risk += 0.01 * max(0, chol - 200) / 50
        risk -= 0.02 * (hdl - 50) / 20
        risk += 0.02 * max(0, bmi - 25) / 5
        if smoke: risk += 0.08
        if self.stress > 0.7: risk += 0.03
        risk = np.clip(risk, 0.01, 0.95)
        return risk * 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_action = 0  # Reset action display

        self.age = self.np_random.uniform(50, 65)
        self.sbp = self.np_random.uniform(140, 170)
        self.total_chol = self.np_random.uniform(220, 280)
        self.hdl = self.np_random.uniform(35, 50)
        self.bmi = self.np_random.uniform(28, 38)
        self.smoking = self.np_random.uniform(0.4, 1.0)
        self.stress = self.np_random.uniform(0.5, 0.9)
        self.exercise_level = 0.2
        self.diet_quality = 0.3

        self.risk_score = self._calculate_risk()
        self.prev_risk = self.risk_score

        self.age_norm = (self.age - 40) / 40
        self.bp_norm = (self.sbp - 90) / 110
        self.chol_norm = (self.total_chol - 100) / 200
        self.hdl_norm = (self.hdl - 20) / 80
        self.bmi_norm = (self.bmi - 18) / 27

        self.risk_history = [self.risk_score]  # ← for graph
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        self.last_action = action  # ← THIS LINE IS CRITICAL FOR UI

        # === All your action effects (unchanged) ===
        if action <= 3:
            intensity = [0, 0.33, 0.66, 1.0][action]
            self.exercise_level = 0.7 * self.exercise_level + 0.3 * intensity
            self.bmi -= 0.08 * intensity
            self.sbp -= 1.2 * intensity
            self.hdl += 0.8 * intensity
        elif action <= 6:
            quality = [0.2, 0.5, 0.9][action-4]
            self.diet_quality = 0.8 * self.diet_quality + 0.2 * quality
            self.total_chol -= 5 * (quality - 0.5)
            self.bmi -= 0.15 * (quality - 0.5)
        elif action <= 8:
            if action == 7:
                self.sbp -= 8
                self.total_chol -= 10
        elif action <= 10:
            good_sleep = 1 if action == 10 else 0
            self.stress -= 0.15 * good_sleep
        elif action == 11:
            self.stress -= 0.25

        # Natural drift
        self.sbp += self.np_random.uniform(-2, 3)
        self.bmi += self.np_random.uniform(-0.05, 0.1)
        self.stress = np.clip(self.stress + self.np_random.uniform(-0.05, 0.1), 0, 1)

        # Clamping
        self.sbp = np.clip(self.sbp, 90, 200)
        self.bmi = np.clip(self.bmi, 18, 45)
        self.hdl = np.clip(self.hdl, 20, 100)
        self.total_chol = np.clip(self.total_chol, 100, 300)
        self.stress = np.clip(self.stress, 0, 1)

        # Update normalized values
        self.bp_norm = (self.sbp - 90) / 110
        self.bmi_norm = (self.bmi - 18) / 27
        self.hdl_norm = (self.hdl - 20) / 80
        self.chol_norm = (self.total_chol - 100) / 200

        old_risk = self.risk_score
        self.risk_score = self._calculate_risk()
        self.risk_history.append(self.risk_score)

        reward = (old_risk - self.risk_score) * 10
        if self.risk_score > 50: reward -= 50
        if self.sbp > 180 or self.bmi > 42: reward -= 20

        terminated = self.current_step >= self.max_steps or self.risk_score > 70
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        from .rendering import draw_screen

        if self.render_mode == "human":
            if not hasattr(self, "screen"):
                self.screen = pygame.display.set_mode((1100, 720))
                pygame.display.set_caption("Cardiovascular Risk Prevention Agent – Ganza Owen Yhaan")
            draw_screen(self, self.screen)
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            screen = pygame.Surface((1100, 720))
            draw_screen(self, screen)
            return pygame.surfarray.array3d(screen)

    def close(self):
        if hasattr(self, "screen"):
            pygame.display.quit()
            pygame.quit()