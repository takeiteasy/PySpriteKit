import spritekit as sk

@sk.main
class TestScene(sk.Scene):
    def enter(self):
        self.camera.position = (320, 240)
        self.camera.rotation = 45
        self.camera.zoom = 2

        test = sk.Texture.checkered(500, 500, 50)
        ass = sk.SpriteActor(texture="assets/textures/Sprite-0001.json",
                             position=(0, 0))

        self.add(sk.SpriteActor(position=(0, 0),
                                texture=test))
        self.add(sk.LineActor(position=(0, 0),
                              end=(-100, -100),
                              color=(1., 0., 0., 1.),
                              thickness=10))
        self.add(sk.RectActor(position=(0, 0),
                              size=(100, 100),
                              color=(0., 1., 0., 1.)))
        self.add(sk.CircleActor(position=(0, 0),
                                radius=100,
                                color=(0., 0., 1., 1.)))
        self.add(sk.EllipseActor(position=(0, 0),
                                 width=200,
                                 height=50,
                                 color=(0., 1., 1., 1.)))
        self.add(sk.PolygonActor(points=((0, 0), (100, 0), (50, 100)),
                                 color=(1., 0., 1., 1.)))
        self.add(sk.LabelActor(text="Hello, world!\nGoodbye, world!",
                               font="ComicMono",
                               font_size=72,
                               align="center",
                               color=(1., 0., 0., 1.)))
        self.add(ass)