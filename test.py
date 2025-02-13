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
        self.add_child(gk.RectangleActor2D(name="test",
                                           width=100,
                                           height=100,
                                           rotation=45.,
                                           color=r.Color(255, 0, 0, 255)))

    def step(self, delta):
        if gk.Keyboard.key_pressed("space"):
            if self.children:
                self.remove_children("test")
            else:
                self.enter()
        