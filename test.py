import quickwindow

from spritekit.cache import *
from spritekit.renderer import *

with quickwindow.quick_window(quit_key=quickwindow.Keys.ESCAPE) as wnd:
    renderer = Renderer()
    horse_texture = load_texture("horse")
    pear_texture = load_texture("pear")
    for delta, events in wnd.loop():
        renderer.draw(rect_vertices(0, 0, 100, 100, color=(1, 0, 0, 1)))
        renderer.draw(rect_vertices(0, 0, 64, 48, scale=4., clip=(0, 0, 64, 48), texture_size=horse_texture.size), texture=horse_texture)
        renderer.draw(rect_vertices(-10, 0, 40, 40, scale=2.), texture=pear_texture)
        renderer.draw(rect_vertices(0, 0, 50, 50, color=(0, 1, 0, 1)))
        renderer.flush()