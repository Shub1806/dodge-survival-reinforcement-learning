from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, LineSegs

from game_core import (
    DodgeGame,
    ARENA,
    ACTION_STAY,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
)
from render_util import make_square, place, ARENA_SCREEN

# How many simulation ticks to run per real-time second.
TICKS_PER_SECOND = 60.0


class DodgeApp(ShowBase):
    def __init__(self):
        super().__init__()
        self.setBackgroundColor(0.05, 0.05, 0.08)
        self.disableMouse()

        self._draw_arena_border()
        self.player_node = make_square(self.aspect2d, 0.5, (0.3, 0.8, 1.0, 1.0))
        self.obstacle_nodes = []

        self.score_text = OnscreenText(
            text="", pos=(-1.25, 0.9), scale=0.07,
            fg=(1, 1, 1, 1), align=TextNode.ALeft, mayChange=True,
        )
        self.message_text = OnscreenText(
            text="", pos=(0, 0), scale=0.09,
            fg=(1, 0.5, 0.4, 1), align=TextNode.ACenter, mayChange=True,
        )

        self.keys = {"up": False, "down": False, "left": False, "right": False}
        bindings = {
            "arrow_up": "up", "w": "up",
            "arrow_down": "down", "s": "down",
            "arrow_left": "left", "a": "left",
            "arrow_right": "right", "d": "right",
        }
        for key, name in bindings.items():
            self.accept(key, self._set_key, [name, True])
            self.accept(f"{key}-up", self._set_key, [name, False])
        self.accept("r", self._restart)
        self.accept("escape", self.userExit)

        self.game = DodgeGame()
        self.game.reset()
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

    def _set_key(self, name, value):
        self.keys[name] = value

    def _current_action(self):
        # Vertical takes priority over horizontal when both are pressed.
        if self.keys["up"]:
            return ACTION_UP
        if self.keys["down"]:
            return ACTION_DOWN
        if self.keys["left"]:
            return ACTION_LEFT
        if self.keys["right"]:
            return ACTION_RIGHT
        return ACTION_STAY

    def _restart(self):
        self.game.reset()
        self.message_text.setText("")

    def _sync_obstacles(self):
        obstacles = self.game.state.obstacles
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

    def _update(self, task):
        dt = globalClock.getDt()
        if self.game.state.alive:
            self.accumulator += dt
            step_time = 1.0 / TICKS_PER_SECOND
            while self.accumulator >= step_time and self.game.state.alive:
                self.game.step(self._current_action())
                self.accumulator -= step_time

        place(self.player_node, self.game.state.player_x, self.game.state.player_y)
        self._sync_obstacles()
        self.score_text.setText(f"Score: {self.game.state.ticks}")
        if not self.game.state.alive:
            self.message_text.setText("Game Over - press R to restart")
        return task.cont


if __name__ == "__main__":
    DodgeApp().run()
