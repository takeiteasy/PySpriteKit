import gamekit as gk

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
        pass

    def step(self, delta):
        pass

    def draw(self):
        super().draw()