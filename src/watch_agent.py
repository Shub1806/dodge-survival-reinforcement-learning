import argparse

from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, LineSegs
from stable_baselines3 import PPO

from dodge_env import DodgeEnv
from render_util import make_square, place, ARENA_SCREEN

TICKS_PER_SECOND = 60.0


class WatchApp(ShowBase):
    def __init__(self, model_path):
        super().__init__()
        self.setBackgroundColor(0.05, 0.05, 0.08)
        self.disableMouse()

        self.env = DodgeEnv()
        self.obs, _ = self.env.reset()
        self.model = PPO.load(model_path)

        self._draw_arena_border()
        self.player_node = make_square(self.aspect2d, 0.5, (0.4, 1.0, 0.6, 1.0))
        self.obstacle_nodes = []

        self.score_text = OnscreenText(
            text="", pos=(-1.25, 0.9), scale=0.07,
            fg=(1, 1, 1, 1), align=TextNode.ALeft, mayChange=True,
        )
        self.accept("escape", self.userExit)

        self.accumulator = 0.0
        self.taskMgr.add(self._update, "update")

    def _draw_arena_border(self):
        b = ARENA_SCREEN
        segs = LineSegs()
        segs.setColor(0.4, 0.4, 0.5, 1)
        segs.setThickness(2)
        for x, y in [(-b, -b), (b, -b), (b, b), (-b, b), (-b, -b)]:
            segs.drawTo(x, 0, y)
        self.aspect2d.attachNewNode(segs.create())

    def _sync_obstacles(self):
        obstacles = self.env.game.state.obstacles
        while len(self.obstacle_nodes) < len(obstacles):
            self.obstacle_nodes.append(
                make_square(self.aspect2d, 0.7, (1.0, 0.4, 0.4, 1.0))
            )
        for i, node in enumerate(self.obstacle_nodes):
            if i < len(obstacles):
                node.show()
                place(node, obstacles[i].x, obstacles[i].y)
            else:
                node.hide()

    def _step_agent(self):
        action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, _, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            self.obs, _ = self.env.reset()

    def _update(self, task):
        dt = globalClock.getDt()
        self.accumulator += dt
        step_time = 1.0 / TICKS_PER_SECOND
        while self.accumulator >= step_time:
            self._step_agent()
            self.accumulator -= step_time

        state = self.env.game.state
        place(self.player_node, state.player_x, state.player_y)
        self._sync_obstacles()
        self.score_text.setText(f"Agent score: {state.ticks}")
        return task.cont


def parse_args():
    p = argparse.ArgumentParser(description="Watch a trained agent play.")
    p.add_argument("--model", type=str, default="models/dodge_ppo",
                   help="path to the saved model (without .zip)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    WatchApp(args.model).run()
