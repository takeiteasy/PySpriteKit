import gamekit as k

@k.main_scene
class TestScene(k.Scene):
    window_attrs = {
        "width": 800,
        "height": 600,
        "title": "Test",
        "exit_key": k.KeyboardKey.KEY_ESCAPE,
        "flags": k.ConfigFlags.FLAG_WINDOW_RESIZABLE,
        "fps": 60
    }
    
    def enter(self):
        pass

    def step(self, delta):
        pass
