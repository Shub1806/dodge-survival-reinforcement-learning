from panda3d.core import CardMaker

from game_core import ARENA

# Scale world units down so the arena fits comfortably on screen.
SCREEN_SCALE = 0.09


def world_to_screen(x, y):
    """Map a world coordinate to Panda3D 2D space (x across, z up)."""
    return x * SCREEN_SCALE, y * SCREEN_SCALE


def make_square(parent, radius, color):
    """Create a centred coloured square sized to a world-space radius."""
    half = radius * SCREEN_SCALE
    maker = CardMaker("square")
    maker.setFrame(-half, half, -half, half)
    node = parent.attachNewNode(maker.generate())
    node.setColor(*color)
    return node


def place(node, x, y):
    sx, sz = world_to_screen(x, y)
    node.setPos(sx, 0, sz)


ARENA_SCREEN = ARENA * SCREEN_SCALE
