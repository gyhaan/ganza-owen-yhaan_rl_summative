# environment/rendering.py  ← FINAL PROFESSIONAL HEART (BEST ONE EVER)
import pygame
import math

pygame.font.init()

# Fonts
TITLE = pygame.font.SysFont("Segoe UI", 40, bold=True)
HEADER = pygame.font.SysFont("Segoe UI", 30, bold=True)
TEXT = pygame.font.SysFont("Segoe UI", 24)
SMALL = pygame.font.SysFont("Segoe UI", 20)

# Colors
BG = (10, 20, 40)
CARD = (20, 30, 55)
GREEN = (0, 255, 150)
YELLOW = (255, 230, 0)
RED = (255, 70, 70)
WHITE = (255, 255, 255)
GRAY = (100, 100, 120)

W, H = 1100, 720

def draw_beating_heart(surface, cx, cy, base_size, health_ratio):
    # health_ratio: 0.0 (critical) → 1.0 (perfect)
    if health_ratio > 0.7:
        color = GREEN
    elif health_ratio > 0.4:
        color = YELLOW
    else:
        color = RED

    # Pulsing size (beat effect)
    pulse = 0.9 + 0.15 * math.sin(pygame.time.get_ticks() * 0.008)
    size = int(base_size * pulse)

    # Real heart shape using curves and circles
    heart_points = []
    for angle in range(0, 360, 5):
        rad = math.radians(angle)
        if angle < 180:
            r = size * (1 - math.sin(rad))
        else:
            r = size * 0.8
        x = cx + r * math.cos(rad)
        y = cy - r * math.sin(rad) * 0.8 + size * 0.1
        heart_points.append((x, y))

    # Draw filled heart
    pygame.draw.polygon(surface, color, heart_points)
    # Glow outline
    pygame.draw.polygon(surface, WHITE, heart_points, max(3, int(4 * pulse)))

    # ECG-style pulse line (optional cool effect)
    if pulse > 1.0:
        pygame.draw.circle(surface, WHITE, (cx + size//2, cy - size//4), 6)

def draw_gauge(surface, x, y, value, label):
    w, h = 500, 36
    pygame.draw.rect(surface, GRAY, (x, y, w, h), border_radius=18)
    fill = int(w * value)
    col = GREEN if value > 0.6 else YELLOW if value > 0.3 else RED
    pygame.draw.rect(surface, col, (x, y, fill, h), border_radius=18)
    txt = SMALL.render(label, True, WHITE)
    surface.blit(txt, (x + 10, y - 28))

def draw_screen(env, screen):
    screen.fill(BG)

    # Title
    title = TITLE.render("Cardiovascular Risk Prevention Agent", True, GREEN)
    screen.blit(title, (W//2 - title.get_width()//2, 20))

    # === LEFT SIDE: BIG BEAUTIFUL HEART ===
    pygame.draw.rect(screen, CARD, (40, 100, 420, 520), border_radius=25)
    
    # Health ratio (0.0 to 1.0)
    health = max(0.0, min(1.0, (50 - env.risk_score) / 40))  # 50% → 0.0, 10% → 1.0
    draw_beating_heart(screen, 250, 340, 160, health)

    # Risk display
    risk_text = HEADER.render(f"{env.risk_score:.2f}%", True, 
                              RED if env.risk_score > 20 else YELLOW if env.risk_score > 12 else GREEN)
    screen.blit(risk_text, (250 - risk_text.get_width()//2, 500))

    # Stats
    stats = [
        f"Week {env.current_step}/520",
        f"Age: {env.age:.0f}",
        f"BP: {env.sbp:.0f} mmHg",
        f"BMI: {env.bmi:.1f}",
        f"Cholesterol: {env.total_chol:.0f}",
        f"HDL: {env.hdl:.0f}",
        f"Stress: {env.stress:.2f}",
    ]
    for i, s in enumerate(stats):
        color = WHITE if i != 0 else GREEN
        txt = TEXT.render(s, True, color)
        screen.blit(txt, (70, 120 + i*60))

    # === RIGHT SIDE: GAUGES ===
    y = 130
    draw_gauge(screen, 500, y, health, "Overall Health")
    y += 70
    draw_gauge(screen, 500, y, env.hdl/100, "HDL (Good Cholesterol)")
    y += 70
    draw_gauge(screen, 500, y, max(0, (180 - env.sbp)/90), "Blood Pressure")
    y += 70
    draw_gauge(screen, 500, y, max(0, (35 - env.bmi)/17), "Weight Control")
    y += 70
    draw_gauge(screen, 500, y, 1.0 - env.stress, "Stress Level")

    # Current action
    actions = ["No Exercise","Light","Moderate","Intense","Poor Diet","Avg Diet","Healthy Diet",
               "Skip Meds","Take Meds","Poor Sleep","Good Sleep","No Stress Relief"]
    action = actions[env.last_action] if 0 <= env.last_action < 12 else "Unknown"
    act_txt = HEADER.render(f"Action: {action}", True, YELLOW)
    screen.blit(act_txt, (500, 70))

    # Risk trend line
    if len(env.risk_history) > 10:
        points = [(60 + i*4, 680 - int(r*1.8)) for i, r in enumerate(env.risk_history[-240:])]
        pygame.draw.lines(screen, GREEN if env.risk_score < env.risk_history[0] else RED, False, points, 4)

    pygame.display.flip()