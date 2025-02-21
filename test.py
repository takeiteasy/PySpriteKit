import spritekit as sk
from typing import override

@sk.main_scene
class TestScene(sk.Scene):
    config = {
        "width": 800,
        "height": 600,
        "title": "Test",
        "exit_key": sk.Keys.escape,
        "flags": sk.Flags.window_resizable,
        "fps": 60
    }

    def add_stuff(self):
        self.add_child(sk.RectangleNode(name="test",
                                        width=100,
                                        height=100,
                                        color=sk.Color(1., 0, 0)))
        self.add_child(sk.CircleNode(name="test",
                                     position=sk.Vector2([100, 100]),
                                     radius=100,
                                     color=sk.Color(0, 1., 0)))
        self.add_child(sk.TriangleNode(name="test",
                                       position2=sk.Vector2([100, 200]),
                                       position3=sk.Vector2([200, 100]),
                                       color=sk.Color(0, 0, 1.)))
        self.add_child(sk.SpriteNode(name="test",
                                     texture=sk.Texture(f"assets/textures/LA.png"),
                                     origin=sk.Vector2([1., 1.]),
                                     scale=sk.Vector2([0.5, 0.5])))
    
    def add_circle(self):
        self.add_child(sk.CircleNode(name="test",
                                     radius=50,
                                     color=sk.Color(1., 0, 1.)))

    @override
    def enter(self):
        self.add_child(sk.LabelNode(name="tset",
                                    text="Hello, World!",
                                    font_size=24,
                                    color=sk.Color(1., 0., 1.)))
        self.add_child(sk.MusicNode(name="bg",
                                    music=sk.Music(f"assets/audio/country.mp3"),
                                    autostart=True))
        self.add_child(sk.TimerNode(name="timer",
                                    interval=1.,
                                    repeat=True,
                                    on_complete=lambda: self.add_circle()))
        self.add_child(sk.RectangleNode(name="poo",
                                        width=50,
                                        height=50,
                                        color=sk.Color(.5, .5, 0)))
        self.add_child(sk.ActionSequence([sk.ActionNode(target=250.,
                                                        easing_fn=sk.ease_bounce_in_out,
                                                        field="position.y",
                                                        actor=self.find_child("poo")),
                                          sk.WaitAction(duration=1.),
                                          sk.ActionNode(target=0.,
                                                        easing_fn=sk.ease_bounce_in_out,
                                                        field="position.y",
                                                        actor=self.find_child("poo"))]))
        self.add_stuff()

    @override
    def step(self, delta):
        if sk.Keyboard.key_pressed("r"):
            if self.find_children("test"):
                self.remove_children("test")
            else:
                self.add_stuff()
        if sk.Keyboard.key_pressed("space"):
            for child in self.find_children("bg"):
                child.toggle()
        
        for child in self.children("test"):
            child.position.x += 100 * delta
        for child in self.children("tset"):
            child.rotation += 100 * delta
        super().step(delta)