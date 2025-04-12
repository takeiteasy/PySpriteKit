import spritekit

@spritekit.main
class TestScene(spritekit.Scene):
    def enter(self):
        self.horse_texture = spritekit.load_texture("horse")
        self.pear_texture = spritekit.load_texture("pear")
    
    def draw(self):
        self.renderer.draw(spritekit.rect_vertices(0, 0, 100, 100, color=(1, 0, 0, 1)))
        self.renderer.draw(spritekit.rect_vertices(0, 0, 64, 48, scale=4., clip=(0, 0, 64, 48), texture_size=self.horse_texture.size), texture=self.horse_texture)
        self.renderer.draw(spritekit.rect_vertices(-10, 0, 40, 40, scale=2.), texture=self.pear_texture)
        self.renderer.draw(spritekit.rect_vertices(0, 0, 50, 50, color=(0, 1, 0, 1)))