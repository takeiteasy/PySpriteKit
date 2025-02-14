import gamekit as gk
from typing import override

@gk.main_scene
class TestScene(gk.Scene):
    config = {
        "width": 800,
        "height": 600,
        "title": "Test",
        "exit_key": gk.KeyboardKey.KEY_ESCAPE,
        "flags": gk.ConfigFlags.FLAG_WINDOW_RESIZABLE,
        "fps": 60
    }
    
    @override
    def enter(self):
        self.add_child(gk.RectangleNode(name="test",
                                        width=100,
                                        height=100,
                                        color=gk.Color(1., 0, 0)))
        self.add_child(gk.CircleNode(name="test",
                                     position=gk.Vector2([100, 100]),
                                     radius=100,
                                     color=gk.Color(0, 1., 0)))
        self.add_child(gk.TriangleNode(name="test",
                                       position2=gk.Vector2([100, 200]),
                                       position3=gk.Vector2([200, 100]),
                                       color=gk.Color(0, 0, 1.)))
        self.add_child(gk.SpriteNode(name="test",
                                     texture=gk.Texture(f"assets/textures/LA.png"),
                                     origin=gk.Vector2([1., 1.]),
                                     scale=gk.Vector2([0.5, 0.5])))
        self.add_child(gk.LabelNode(text="Hello, World!",
                                    font_size=24,
                                    color=gk.Color(1., 0., 1.)))
        self.add_child(gk.MusicNode(music=gk.Music(f"assets/audio/country.mp3"),
                                    autostart=True))

    @override
    def step(self, delta):
        if gk.Keyboard.key_pressed("space"):
            if self.find_children("test"):
                self.remove_children("test")
            else:
                self.enter()
        
        for child in self.children("test"):
            child.position.x += 100 * delta
        for child in self.children("tset"):
            child.rotation += 100 * delta
        super().step(delta)