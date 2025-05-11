import spritekit as sk

@sk.main
class TestScene(sk.Scene):
    def enter(self):
        poo = sk.AnimatedSpriteNode(atlas="Sprite-0001",
                                    initial_animation="right")
        self.add(poo)
