import pygame
import numpy as np
from typing import Dict, Optional
import math

class HealthEnvRenderer:
    """
    Pygame-based visualization for the Heart Health Environment
    Shows patient avatar, health metrics, interventions, and risk timeline
    """
    
    def __init__(self, screen_width: int = 1000, screen_height: int = 700):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Heart Health Intervention Agent")
        
        # Colors
        self.BACKGROUND = (240, 245, 255)
        self.PANEL_BG = (255, 255, 255)
        self.TEXT_COLOR = (50, 50, 80)
        self.HIGHLIGHT = (41, 128, 185)
        self.WARNING = (231, 76, 60)
        self.SAFE = (39, 174, 96)
        self.GAUGE_BG = (220, 220, 220)
        
        # Fonts
        self.title_font = pygame.font.SysFont('arial', 28, bold=True)
        self.header_font = pygame.font.SysFont('arial', 20, bold=True)
        self.normal_font = pygame.font.SysFont('arial', 16)
        self.small_font = pygame.font.SysFont('arial', 14)
        
        # Patient avatar states
        self.avatar_states = []
        self.max_history = 50
        
        # Load assets (placeholder - you can replace with actual images)
        self.assets_loaded = False
        self.load_assets()
        
        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()
    
    def load_assets(self):
        """Load or create visual assets"""
        try:
            # Placeholder - you can add actual images to demos/assets/
            self.patient_avatar = self.create_placeholder_avatar()
            self.heart_icon = self.create_heart_icon()
            self.assets_loaded = True
        except:
            self.assets_loaded = False
    
    def create_placeholder_avatar(self):
        """Create a simple placeholder avatar"""
        surf = pygame.Surface((120, 160), pygame.SRCALPHA)
        # Head
        pygame.draw.ellipse(surf, (255, 218, 185), (30, 10, 60, 70))
        # Body
        pygame.draw.rect(surf, (70, 130, 180), (25, 80, 70, 80))
        return surf
    
    def create_heart_icon(self):
        """Create a heart icon"""
        surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        # Simple heart shape
        points = [
            (20, 35), (10, 25), (10, 15), (20, 10), 
            (30, 15), (30, 25), (20, 35)
        ]
        pygame.draw.polygon(surf, (231, 76, 60), points)
        return surf
    
    def reset(self):
        """Reset renderer state"""
        self.avatar_states = []
    
    def render(self, health_state: Dict, action: Optional[np.ndarray], reward: float):
        """Main rendering function"""
        self.screen.fill(self.BACKGROUND)
        
        # Draw main panels
        self.draw_patient_panel(health_state)
        self.draw_metrics_panel(health_state)
        self.draw_intervention_panel(action)
        self.draw_risk_timeline(health_state)
        self.draw_reward_panel(reward)
        
        # Add instructions
        self.draw_instructions()
        
        pygame.display.flip()
        
        # Handle events to prevent freezing and allow window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        
        # Control frame rate - wait for a bit to make it visible
        self.clock.tick(2)  # 2 FPS so you can see the changes
        
        return True
    
    def draw_instructions(self):
        """Draw instructions for the user"""
        instructions = [
            "Press ESC to close window",
            "Window will auto-close after all steps"
        ]
        
        y_pos = self.screen_height - 50
        for instruction in instructions:
            text = self.small_font.render(instruction, True, (100, 100, 100))
            self.screen.blit(text, (50, y_pos))
            y_pos += 20
    
    def draw_patient_panel(self, health_state: Dict):
        """Draw patient avatar and basic info"""
        panel_rect = pygame.Rect(50, 50, 300, 200)
        pygame.draw.rect(self.screen, self.PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, self.HIGHLIGHT, panel_rect, 2, border_radius=12)
        
        # Title
        title = self.title_font.render("Virtual Patient", True, self.TEXT_COLOR)
        self.screen.blit(title, (panel_rect.x + 20, panel_rect.y + 15))
        
        # Avatar
        avatar_x = panel_rect.x + 30
        avatar_y = panel_rect.y + 60
        self.screen.blit(self.patient_avatar, (avatar_x, avatar_y))
        
        # Update avatar color based on health
        risk_color = self.get_risk_color(health_state['risk_score'])
        pygame.draw.circle(self.screen, risk_color, 
                          (avatar_x + 60, avatar_y + 35), 8)
        
        # Basic info
        info_y = avatar_y + 20
        age_text = self.normal_font.render(f"Age: {int(health_state['age'])}", True, self.TEXT_COLOR)
        smoking_status = "Smoker" if health_state['smoking_status'] == 1 else "Non-smoker"
        smoke_text = self.normal_font.render(f"Status: {smoking_status}", True, self.TEXT_COLOR)
        
        self.screen.blit(age_text, (avatar_x + 130, info_y))
        self.screen.blit(smoke_text, (avatar_x + 130, info_y + 30))
    
    def draw_metrics_panel(self, health_state: Dict):
        """Draw health metrics with gauges"""
        panel_rect = pygame.Rect(50, 270, 300, 380)
        pygame.draw.rect(self.screen, self.PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, self.HIGHLIGHT, panel_rect, 2, border_radius=12)
        
        # Title
        title = self.header_font.render("Health Metrics", True, self.TEXT_COLOR)
        self.screen.blit(title, (panel_rect.x + 20, panel_rect.y + 15))
        
        metrics = [
            ("Blood Pressure", f"{int(health_state['systolic_bp'])}/{int(health_state['diastolic_bp'])} mmHg", 
             health_state['systolic_bp'], 90, 180),
            ("Cholesterol", f"{int(health_state['cholesterol'])} mg/dL", 
             health_state['cholesterol'], 150, 300),
            ("Weight", f"{int(health_state['weight'])} kg", 
             health_state['weight'], 50, 150),
            ("Stress Level", f"{health_state['stress_level']:.1f}/10", 
             health_state['stress_level'], 0, 10),
        ]
        
        y_offset = 60
        for name, value, current, min_val, max_val in metrics:
            self.draw_gauge(panel_rect.x + 30, panel_rect.y + y_offset, 
                          250, 25, name, value, current, min_val, max_val)
            y_offset += 65
    
    def draw_gauge(self, x: int, y: int, width: int, height: int, 
                  name: str, value: str, current: float, min_val: float, max_val: float):
        """Draw a health metric gauge"""
        # Gauge background
        gauge_rect = pygame.Rect(x, y + 25, width, height)
        pygame.draw.rect(self.screen, self.GAUGE_BG, gauge_rect, border_radius=5)
        
        # Calculate fill percentage
        fill_percent = (current - min_val) / (max_val - min_val)
        fill_width = max(5, int(width * fill_percent))
        
        # Determine color based on value
        if "Blood Pressure" in name:
            fill_color = self.WARNING if current > 140 else self.SAFE
        elif "Cholesterol" in name:
            fill_color = self.WARNING if current > 240 else self.SAFE
        elif "Weight" in name:
            fill_color = self.WARNING if current > 100 else self.SAFE
        elif "Stress" in name:
            fill_color = self.WARNING if current > 7 else self.SAFE
        else:
            fill_color = self.HIGHLIGHT
        
        # Fill gauge
        fill_rect = pygame.Rect(x, y + 25, fill_width, height)
        pygame.draw.rect(self.screen, fill_color, fill_rect, border_radius=5)
        
        # Labels
        name_text = self.normal_font.render(name, True, self.TEXT_COLOR)
        value_text = self.normal_font.render(value, True, self.TEXT_COLOR)
        
        self.screen.blit(name_text, (x, y))
        self.screen.blit(value_text, (x + width - 80, y))
    
    def draw_intervention_panel(self, action: Optional[np.ndarray]):
        """Draw current interventions"""
        panel_rect = pygame.Rect(370, 50, 280, 200)
        pygame.draw.rect(self.screen, self.PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, self.HIGHLIGHT, panel_rect, 2, border_radius=12)
        
        title = self.header_font.render("Current Interventions", True, self.TEXT_COLOR)
        self.screen.blit(title, (panel_rect.x + 20, panel_rect.y + 15))
        
        if action is None:
            no_action = self.normal_font.render("No actions taken", True, self.TEXT_COLOR)
            self.screen.blit(no_action, (panel_rect.x + 30, panel_rect.y + 60))
            return
        
        interventions = [
            ("Exercise", ["Sedentary", "Moderate", "Vigorous"]),
            ("Diet", ["Poor", "Balanced", "Optimal"]),
            ("Medication", ["Skipped", "Taken"]),
            ("Sleep", ["Poor", "Adequate", "Optimal"]),
            ("Stress Mgmt", ["None", "Moderate", "High"]),
        ]
        
        y_offset = 50
        for i, (name, levels) in enumerate(interventions):
            action_level = action[i] if i < len(action) else 0
            level_name = levels[action_level] if action_level < len(levels) else "Unknown"
            
            text = self.normal_font.render(f"{name}: {level_name}", True, self.TEXT_COLOR)
            self.screen.blit(text, (panel_rect.x + 30, panel_rect.y + y_offset))
            y_offset += 30
    
    def draw_risk_timeline(self, health_state: Dict):
        """Draw risk score timeline"""
        panel_rect = pygame.Rect(370, 270, 580, 200)
        pygame.draw.rect(self.screen, self.PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, self.HIGHLIGHT, panel_rect, 2, border_radius=12)
        
        title = self.header_font.render("Heart Disease Risk Timeline", True, self.TEXT_COLOR)
        self.screen.blit(title, (panel_rect.x + 20, panel_rect.y + 15))
        
        # Add current risk to history
        self.avatar_states.append(health_state['risk_score'])
        if len(self.avatar_states) > self.max_history:
            self.avatar_states.pop(0)
        
        # Draw timeline
        timeline_rect = pygame.Rect(panel_rect.x + 30, panel_rect.y + 60, 520, 100)
        pygame.draw.rect(self.screen, self.GAUGE_BG, timeline_rect, border_radius=5)
        
        if len(self.avatar_states) > 1:
            points = []
            for i, risk in enumerate(self.avatar_states):
                x = timeline_rect.x + (i / (len(self.avatar_states) - 1)) * timeline_rect.width
                y = timeline_rect.y + timeline_rect.height - (risk / 100) * timeline_rect.height
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.WARNING, False, points, 3)
        
        # Risk labels
        low_risk = self.small_font.render("Low Risk", True, self.SAFE)
        high_risk = self.small_font.render("High Risk", True, self.WARNING)
        current_risk = self.normal_font.render(f"Current Risk: {health_state['risk_score']:.1f}%", 
                                             True, self.get_risk_color(health_state['risk_score']))
        
        self.screen.blit(low_risk, (timeline_rect.x, timeline_rect.y - 20))
        self.screen.blit(high_risk, (timeline_rect.x + timeline_rect.width - 70, timeline_rect.y - 20))
        self.screen.blit(current_risk, (panel_rect.x + 30, panel_rect.y + 170))
    
    def draw_reward_panel(self, reward: float):
        """Draw reward feedback"""
        panel_rect = pygame.Rect(670, 50, 280, 200)
        pygame.draw.rect(self.screen, self.PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, self.HIGHLIGHT, panel_rect, 2, border_radius=12)
        
        title = self.header_font.render("Performance", True, self.TEXT_COLOR)
        self.screen.blit(title, (panel_rect.x + 20, panel_rect.y + 15))
        
        # Reward value with color coding
        reward_color = self.SAFE if reward >= 0 else self.WARNING
        reward_text = self.title_font.render(f"Reward: {reward:+.1f}", True, reward_color)
        
        # Interpretation
        if reward > 5:
            feedback = "Excellent health improvement!"
        elif reward > 0:
            feedback = "Positive health trend"
        elif reward > -2:
            feedback = "Needs improvement"
        else:
            feedback = "Health deterioration detected"
        
        feedback_text = self.normal_font.render(feedback, True, self.TEXT_COLOR)
        
        # Heart icon
        self.screen.blit(self.heart_icon, (panel_rect.x + 120, panel_rect.y + 70))
        
        self.screen.blit(reward_text, (panel_rect.x + 50, panel_rect.y + 120))
        self.screen.blit(feedback_text, (panel_rect.x + 30, panel_rect.y + 160))
    
    def get_risk_color(self, risk_score: float):
        """Get color based on risk level"""
        if risk_score < 20:
            return self.SAFE
        elif risk_score < 40:
            return (255, 204, 0)  # Yellow
        elif risk_score < 60:
            return (255, 136, 0)  # Orange
        else:
            return self.WARNING
    
    def close(self):
        """Clean up Pygame resources"""
        pygame.quit()