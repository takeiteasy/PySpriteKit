import spritekit

@spritekit.main
class TestScene(spritekit.Scene):
    def enter(self):
        self.add(spritekit.Sprite(position=(0, 0), texture="pear"))
        self.add(spritekit.Line(position=(0, 0), end=(-100, -100), color=(1., 0., 0., 1.), thickness=10))
        self.add(spritekit.Rect(position=(0, 0), size=(100, 100), color=(0., 1., 0., 1.)))
        self.add(spritekit.Circle(position=(0, 0), diameter=100, color=(0., 0., 1., 1.)))
        self.add(spritekit.Ellipse(position=(0, 0), width=200, height=50, color=(0., 0., 1., 1.)))
        self.add(spritekit.Polygon(position=(0, 0), points=((0, 0), (100, 0), (50, 100)), color=(1., 0., 0., 1.)))
