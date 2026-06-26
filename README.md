# Dodge Survive

A small top-down survival game I made in Python, plus a reinforcement learning
agent that learns to play it by itself.

The game is simple: you move a square around an arena, obstacles come flying in
from the edges, and you try to last as long as you can without getting hit. The
longer you survive the harder it gets. On top of that I trained an AI agent to
play the same game, so you can either play it yourself or watch the agent play.

I built this to actually understand how reinforcement learning works from start
to finish: building an environment, setting up a reward, training an agent, and
watching it slowly get better. Turns out the agent learns purely from trial and
error, it never watches a human play.

## What's in here

```
src/
  game_core.py    # the actual game logic (movement, obstacles, collisions)
  dodge_env.py    # wraps the game so an RL library can train on it
  render_util.py  # helper code for drawing
  play_human.py   # play the game yourself
  train.py        # trains the AI agent
  watch_agent.py  # watch the trained agent play
```

## How it works (short version)

I kept the game logic completely separate from the graphics. The game itself is
just plain Python with no graphics code in it at all. That way the training can
run without drawing anything on screen, which makes it way faster. The graphics
(Panda3D) are only used when a human plays or when you watch the agent.

For the AI part: the agent sees where it is and where the nearest obstacles are,
picks a move (stay / up / down / left / right), and gets a small reward for every
moment it stays alive and a penalty when it dies. It plays the game thousands of
times and gradually figures out that surviving longer is good. I used PPO from
Stable-Baselines3 for the training.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running it

Play it yourself (arrow keys or WASD, R to restart):

```bash
cd src
python play_human.py
```

Train the agent (takes a few minutes, you'll see the average survival time go up
as it learns):

```bash
cd src
python train.py --timesteps 300000 --envs 4 --out models/dodge_ppo
```

Watch the trained agent play:

```bash
cd src
python watch_agent.py --model models/dodge_ppo
```

You only need to run the training once. When it finishes it saves the agent to
the models folder and you're good to go.

## Things I might add later

- Different types of obstacles that move in different ways
- Train it on the raw pixels instead of coordinates
- Try other algorithms (DQN, A2C) and see which one survives longest
- Record training graphs with TensorBoard

## Notes

This was a learning project so the game is intentionally simple. The point was
the RL side, not making a fancy game. The trained model file isn't included in
the repo (it's ignored in .gitignore) so just run the training once to generate
it.