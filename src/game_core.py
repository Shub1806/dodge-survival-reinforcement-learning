import math
import random
from dataclasses import dataclass, field

# Arena spans [-ARENA, ARENA] on both x and y axes.
ARENA = 10.0
PLAYER_RADIUS = 0.5
PLAYER_SPEED = 0.35

OBSTACLE_RADIUS = 0.7
OBSTACLE_SPEED_MIN = 0.16
OBSTACLE_SPEED_MAX = 0.32

# How many of the nearest obstacles the observation reports back.
TRACKED_OBSTACLES = 5

# Difficulty ramp: obstacles spawn every N ticks, and N shrinks over time.
SPAWN_INTERVAL_START = 28
SPAWN_INTERVAL_MIN = 7
SPAWN_RAMP = 0.015

MAX_STEPS = 2000

# Action ids. Kept small so the agent learns quickly.
ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
NUM_ACTIONS = 5

_ACTION_VECTORS = {
    ACTION_STAY: (0.0, 0.0),
    ACTION_UP: (0.0, 1.0),
    ACTION_DOWN: (0.0, -1.0),
    ACTION_LEFT: (-1.0, 0.0),
    ACTION_RIGHT: (1.0, 0.0),
}


@dataclass
class Obstacle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = OBSTACLE_RADIUS


@dataclass
class GameState:
    player_x: float = 0.0
    player_y: float = 0.0
    obstacles: list = field(default_factory=list)
    ticks: int = 0
    spawn_timer: float = 0.0
    alive: bool = True


class DodgeGame:
    """Headless dodge/survive simulation.

    Use it like this:
        game = DodgeGame()
        game.reset()
        state, reward, done = game.step(action)
    """

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.state = GameState()

    def seed(self, seed):
        self.rng = random.Random(seed)

    def reset(self):
        self.state = GameState()
        self.state.spawn_timer = SPAWN_INTERVAL_START
        return self.state

    def _current_spawn_interval(self):
        interval = SPAWN_INTERVAL_START - self.state.ticks * SPAWN_RAMP
        return max(SPAWN_INTERVAL_MIN, interval)

    def _spawn_obstacle(self):
        # Pick a random edge, place the obstacle just outside it, and aim it
        # across the arena toward a jittered point near the centre.
        edge = self.rng.randint(0, 3)
        margin = ARENA + OBSTACLE_RADIUS
        if edge == 0:      # top
            x, y = self.rng.uniform(-ARENA, ARENA), margin
        elif edge == 1:    # bottom
            x, y = self.rng.uniform(-ARENA, ARENA), -margin
        elif edge == 2:    # left
            x, y = -margin, self.rng.uniform(-ARENA, ARENA)
        else:              # right
            x, y = margin, self.rng.uniform(-ARENA, ARENA)

        target_x = self.rng.uniform(-ARENA * 0.5, ARENA * 0.5)
        target_y = self.rng.uniform(-ARENA * 0.5, ARENA * 0.5)
        dx, dy = target_x - x, target_y - y
        dist = math.hypot(dx, dy) or 1.0
        speed = self.rng.uniform(OBSTACLE_SPEED_MIN, OBSTACLE_SPEED_MAX)
        vx, vy = dx / dist * speed, dy / dist * speed
        self.state.obstacles.append(Obstacle(x, y, vx, vy))

    def _move_player(self, action):
        ax, ay = _ACTION_VECTORS.get(action, (0.0, 0.0))
        self.state.player_x += ax * PLAYER_SPEED
        self.state.player_y += ay * PLAYER_SPEED
        limit = ARENA - PLAYER_RADIUS
        self.state.player_x = max(-limit, min(limit, self.state.player_x))
        self.state.player_y = max(-limit, min(limit, self.state.player_y))

    def _advance_obstacles(self):
        bound = ARENA + OBSTACLE_RADIUS * 2
        survivors = []
        for ob in self.state.obstacles:
            ob.x += ob.vx
            ob.y += ob.vy
            if -bound <= ob.x <= bound and -bound <= ob.y <= bound:
                survivors.append(ob)
        self.state.obstacles = survivors

    def _check_collision(self):
        for ob in self.state.obstacles:
            dx = ob.x - self.state.player_x
            dy = ob.y - self.state.player_y
            if math.hypot(dx, dy) <= ob.radius + PLAYER_RADIUS:
                return True
        return False

    def step(self, action):
        """Advance one tick. Returns (state, reward, done)."""
        if not self.state.alive:
            return self.state, 0.0, True

        self._move_player(action)

        self.state.spawn_timer -= 1.0
        if self.state.spawn_timer <= 0:
            self._spawn_obstacle()
            self.state.spawn_timer = self._current_spawn_interval()

        self._advance_obstacles()
        self.state.ticks += 1

        done = False
        reward = 0.1  # small reward for each tick survived

        if self._check_collision():
            self.state.alive = False
            done = True
            reward = -1.0
        elif self.state.ticks >= MAX_STEPS:
            done = True

        return self.state, reward, done

    def nearest_obstacles(self, count=TRACKED_OBSTACLES):
        """Return the `count` obstacles closest to the player, nearest first."""
        ordered = sorted(
            self.state.obstacles,
            key=lambda ob: (ob.x - self.state.player_x) ** 2
            + (ob.y - self.state.player_y) ** 2,
        )
        return ordered[:count]
