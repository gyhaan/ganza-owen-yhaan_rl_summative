
# Cardiovascular Risk Prevention Agent  
**Reinforcement Learning for Personalized 10-Year Heart Disease Prevention**  

## Project Overview
A reinforcement learning agent that acts as an AI clinician: given a high-risk patient, it prescribes the optimal sequence of lifestyle changes and medications over 520 weeks (10 years) to minimize the patient's 10-year ASCVD (cardiovascular disease) risk.

The best agents consistently reduce risk from ~45–60% down to **~9.09%** — far below typical real-world outcomes.


## Quick Start – Live Demo (Best DQN Agent)
```bash
pip install gymnasium stable-baselines3 torch pandas pygame matplotlib seaborn

python demo_best_agent.py
```
You will see:
- Real-time patient dashboard
- Agent actions printed in terminal
- Risk dropping week-by-week from ~50% → ~9.09%

## Reproduce Training
```bash
python training/dqn_training.py  
python training/ppo_training.py
python training/a2c_training.py
python training/reinforce_training.py
```

## Requirements
```
gymnasium
stable-baselines3
torch
pandas
numpy
pygame
matplotlib
seaborn
```
## Video Demonstration
`heart_agent_demo.mp4` – 30-second clip showing the best DQN agent treating a new high-risk patient in real time.

## Conclusion
The project successfully demonstrates that modern reinforcement learning (especially **DQN**) can autonomously learn safe, highly effective, and personalized cardiovascular prevention strategies that dramatically outperform standard guideline-based care in simulation.
