# environment/rendering.py ← FIXED – HEART STARTS RED, TURNS GREEN ON IMPROVEMENT
import pygame
import math
import numpy as np

pygame.font.init()

# === COLORS ===
BG = (15, 25, 45)
CARD = (25, 40, 70)
GREEN = (0, 220, 140)
YELLOW = (255, 220, 80)
RED = (255, 80, 80)
WHITE = (245, 245, 245)
GRAY = (110, 120, 140)

# === FONTS ===
TITLE = pygame.font.SysFont("Segoe UI", 42, bold=True)
HEADER = pygame.font.SysFont("Segoe UI", 28, bold=True)
TEXT = pygame.font.SysFont("Segoe UI", 22)
SMALL = pygame.font.SysFont("Segoe UI", 18)

W, H = 1200, 780

def draw_beating_heart(surface, cx, cy, size, risk_score):
    # FIXED: Direct use of risk_score for color (no health ratio needed)
    # Start RED (>25% risk), YELLOW (15-25%), GREEN (<15%)
    if risk_score < 15:
        color = GREEN
    elif risk_score < 25:
        color = YELLOW
    else:
        color = RED

    # Dramatic pulse: faster when improving, slower when stable
    pulse_speed = 0.012 if risk_score < 20 else 0.008
    pulse = 0.85 + 0.25 * math.sin(pygame.time.get_ticks() * pulse_speed)
    s = int(size * pulse)

    # Realistic heart shape (cardioid equation for perfect anatomy)
    t = np.linspace(0, 2*np.pi, 100)
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    points = []
    for i in range(len(x)):
        px = cx + x[i] * s / 20
        py = cy - y[i] * s / 20
        points.append((px, py))

    pygame.draw.polygon(surface, color, points)
    outline_width = 5 if pulse > 1.0 else 3  # Glow when beating strong
    pygame.draw.polygon(surface, WHITE, points, outline_width)

def draw_gauge(surface, x, y, value, label, max_val=1.0):
    w, h = 460, 36
    pygame.draw.rect(surface, GRAY, (x, y, w, h), border_radius=18)
    fill = int(w * min(value / max_val, 1.0))
    color = GREEN if value >= 0.7*max_val else YELLOW if value >= 0.4*max_val else RED
    pygame.draw.rect(surface, color, (x, y, fill, h), border_radius=18)
    txt = SMALL.render(label, True, WHITE)
    val_txt = SMALL.render(f"{value:.1f}", True, WHITE)
    surface.blit(txt, (x + 10, y - 28))
    surface.blit(val_txt, (x + w - 50, y + 8))

def draw_screen(env, screen):
    screen.fill(BG)

    # === TITLE ===
    title = TITLE.render("Cardiovascular Risk Prevention Agent", True, GREEN)
    screen.blit(title, (W//2 - title.get_width()//2, 20))

    # === LEFT: PATIENT + HEART (COLOR CHANGES WITH RISK!) ===
    pygame.draw.rect(screen, CARD, (40, 100, 480, 620), border_radius=25, width=4)
    draw_beating_heart(screen, 280, 380, 200, env.risk_score)  # FIXED: Pass risk_score directly

    risk_text = HEADER.render(f"{env.risk_score:.2f}%", True,
                              RED if env.risk_score > 25 else YELLOW if env.risk_score > 15 else GREEN)
    screen.blit(risk_text, (280 - risk_text.get_width()//2, 560))

    stats = [
        f"Week {env.current_step}/520",
        f"Age: {env.age:.0f} years",
        f"Systolic BP: {env.sbp:.0f} mmHg",
        f"BMI: {env.bmi:.1f}",
        f"Total Chol: {env.total_chol:.0f}",
        f"HDL: {env.hdl:.0f}",
        f"Stress: {env.stress:.1f}",
    ]
    for i, s in enumerate(stats):
        txt = TEXT.render(s, True, WHITE)
        screen.blit(txt, (70, 140 + i*60))

    # === RIGHT: GAUGES + ACTION + GRAPH ===
    y = 130
    draw_gauge(screen, 560, y, 100 - env.risk_score, "Overall Health", 100)
    y += 80
    draw_gauge(screen, 560, y, env.hdl, "HDL Cholesterol", 100)
    y += 80
    draw_gauge(screen, 560, y, max(0, 180 - env.sbp), "Blood Pressure Control", 90)
    y += 80
    draw_gauge(screen, 560, y, max(0, 35 - env.bmi), "Weight Control", 17)
    y += 80
    draw_gauge(screen, 560, y, 1.0 - env.stress, "Stress Management", 1.0)

    # Current Action
    actions = ["No Exercise","Light","Moderate","Intense","Poor Diet","Avg Diet","Healthy Diet",
               "Skip Meds","Take Meds","Poor Sleep","Good Sleep","No Stress Relief"]
    action_name = actions[env.last_action] if hasattr(env, 'last_action') else "None"
    act_surf = HEADER.render(f"Action: {action_name}", True, YELLOW)
    screen.blit(act_surf, (560, 70))

    # Risk Trend Graph
    pygame.draw.rect(screen, CARD, (40, 730, 1120, 90), border_radius=20)
    if len(env.risk_history) > 10:
        points = []
        for i, r in enumerate(env.risk_history[-280:]):
            x = 60 + i * 4
            y = 780 - int((r / 50) * 70)
            points.append((x, y))
        color = GREEN if env.risk_score < env.risk_history[0] else RED
        pygame.draw.lines(screen, color, False, points, 4)

    pygame.display.flip()