"""
Gymnasium environment that wraps the dodge/survive simulation.

This exposes the game through the standard reset() / step() reinforcement-learning
interface so it can be trained with any RL library. The observation is a flat
vector: the player position plus the relative position and velocity of the
nearest few obstacles. Actions are the five discrete moves from the game core.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game_core import (
    DodgeGame,
    ARENA,
    NUM_ACTIONS,
    TRACKED_OBSTACLES,
    OBSTACLE_SPEED_MAX,
)


class DodgeEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = DodgeGame()
        self.render_mode = render_mode

        # Observation layout:
        #   [player_x, player_y,
        #    (rel_x, rel_y, vx, vy) for each tracked obstacle]
        obs_dim = 2 + TRACKED_OBSTACLES * 4
        # Observations are normalised, so a generous finite bound is enough.
        high = np.full(obs_dim, 5.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    def _build_observation(self):
        s = self.game.state
        obs = [s.player_x / ARENA, s.player_y / ARENA]
        nearest = self.game.nearest_obstacles()
        for i in range(TRACKED_OBSTACLES):
            if i < len(nearest):
                ob = nearest[i]
                obs.extend([
                    (ob.x - s.player_x) / (2 * ARENA),
                    (ob.y - s.player_y) / (2 * ARENA),
                    ob.vx / OBSTACLE_SPEED_MAX,
                    ob.vy / OBSTACLE_SPEED_MAX,
                ])
            else:
                # No obstacle in this slot yet: report it as far away and still.
                obs.extend([1.0, 1.0, 0.0, 0.0])
        return np.asarray(obs, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.seed(seed)
        self.game.reset()
        return self._build_observation(), {}

    def step(self, action):
        _, reward, done = self.game.step(int(action))
        obs = self._build_observation()
        terminated = done and not self.game.state.alive
        truncated = done and self.game.state.alive
        info = {"ticks": self.game.state.ticks}
        return obs, reward, terminated, truncated, info

    def render(self):
        # Rendering is handled by the Panda3D front-ends, not by the env itself.
        return None
