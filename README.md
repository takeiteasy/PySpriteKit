# PySpriteKit

> [!WARNING]
> Work in progress, see [TODO](#todo)

2D scene+actor framework built on top of [raylib](https://github.com/raysan5/raylib), based off [SpriteKit](https://developer.apple.com/documentation/spritekit) by Apple.

> [!NOTE]
> `pip install spritekit==0.0.4`

### Features

- Raylib bindings (using [raylib-python-cffi](https://github.com/electronstudio/raylib-python-cffi/tree/master))
- 2D rendering (shapes, sprites, text)
- Audio (music, sound effects)
- Scene management
- Actor framework
- Linear algebra (vector2,3,4, matrix4, taken from [Pyrr](https://github.com/adamlwgriffiths/Pyrr))
- Easing functions (taken from [raylib-py](https://github.com/overdev/raylib-py/blob/master/src/raylibpy/easings.py))
- Finite state machine (build on top of [transitions](https://github.com/pytransitions/transitions))
- Action framework

### Example

```python3
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

    @override
    def enter(self):
        self.add_child(sk.LabelNode(name="tset",
                                    text="Hello, World!",
                                    font_size=24,
                                    color=sk.Color(1., 0., 1.)))
        self.add_child(sk.MusicNode(name="bg",
                                    music=sk.Music(f"assets/audio/country.mp3"),
                                    autostart=True))
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
        super().step(delta) # update scene internal step
```

## Requirements

```
attrs==25.1.0
cffi==1.17.1
multipledispatch==1.0.0
numpy==2.2.2
pycparser==2.22
pyglsl==0.0.5
raylib==5.5.0.2
six==1.17.0
transitions==0.9.2
typing==3.7.4.3
```

## TODO

- [X] ~~Action Nodes~~
- [X] ~~Timer Node~~
- [ ] RenderTexture wrapper
- [X] ~~Emitter Node~~
- [ ] TileMap Node
- [ ] 3D Nodes
- [ ] 2D + 3D Collision
- [ ] 2D Physics
- [ ] Video Node
- [ ] Transform Node
- [ ] Add examples
- [ ] Add documentation

## LICENSE

```
PySpriteKit

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
