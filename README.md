# gamekit

> [!WARNING]
> Work in progress

```python3
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
        self.add_child(gk.Rectangle(name="test",
                                    width=100,
                                    height=100,
                                    color=gk.Color(1., 0, 0)))
        self.add_child(gk.Circle(name="test",
                                 position=gk.Vector2([100, 100]),
                                 radius=100,
                                 color=gk.Color(0, 1., 0)))
        self.add_child(gk.Triangle(name="test",
                                   position2=gk.Vector2([100, 200]),
                                   position3=gk.Vector2([200, 100]),
                                   color=gk.Color(0, 0, 1.)))
        self.add_child(gk.Sprite(name="test",
                                 texture=gk.Texture(f"assets/textures/LA"),
                                 origin=gk.Vector2([1., 1.]),
                                 scale=gk.Vector2([0.5, 0.5])))

    @override
    def step(self, delta):
        if gk.Keyboard.key_pressed("space"):
            if self.children:
                self.remove_children("test")
            else:
                self.enter()
        
        for child in self.children("test"):
            child.position.x += 100 * delta
```

## LICENSE
```
gamekit

Copyright (C) 2025 George Watson

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
