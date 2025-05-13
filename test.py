import spritekit as sk

@sk.main
class TestScene(sk.Scene):
    def enter(self):
        poo = sk.AtlasNode(atlas="Sprite-0001",
                           initial_animation="left")
        bum = sk.AtlasNode(atlas="Sprite-0001",
                           initial_animation="right",
                           position=(100, 100))
        txt = sk.LabelNode(text="Hello World",
                           font="ComicMono",
                           font_size=24)
        self.add(poo)
        self.add(bum)
        self.add(txt)
