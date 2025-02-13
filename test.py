import gamekit as gk
import pyray as r

@gk.main_scene
class TestScene(gk.Scene):
    window_attrs = {
        "width": 800,
        "height": 600,
        "title": "Test",
        "exit_key": gk.KeyboardKey.KEY_ESCAPE,
        "flags": gk.ConfigFlags.FLAG_WINDOW_RESIZABLE,
        "fps": 60
    }
    
    def enter(self):
        self.add_child(gk.Rectangle2D(name="test",
                                      width=100,
                                      height=100,
                                      color=r.Color(255, 0, 0, 255)))
        self.add_child(gk.Circle2D(name="test",
                                   position=gk.Vector2([100, 100]),
                                   radius=100,
                                   color=r.Color(0, 255, 0, 255)))
        self.add_child(gk.Triangle2D(name="test",
                                     position2=gk.Vector2([100, 100]),
                                     position3=gk.Vector2([200, 100]),
                                     color=r.Color(0, 0, 255, 255)))

    def step(self, delta):
        if gk.Keyboard.key_pressed("space"):
            if self.children:
                self.remove_children("test")
            else:
                self.enter()
        