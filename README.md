# PySpriteKit

> [!WARNING]
> Work in progress, see [TODO](#todo)

2D scene+actor framework, see [test.py](https://github.com/takeiteasy/PySpriteKit/blob/master/test.py) for working example.

```python
import spritekit as sk

@sk.main
class TestScene(sk.Scene):
    def enter(self):
        self.camera.position = (320, 240)
        self.camera.rotation = 45
        self.camera.zoom = .5

        self.add(sk.SpriteActor(position=(0, 0),
                                texture="pear"))
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
```

> [!NOTE]
> `pip install spritekit==0.2.1`

## Features

- [X] Scenes
- [X] Batched renderer
- [X] Easy + Flexible
- [X] Line, Rectangle, Ellipse, Circle, Polygon
- [X] Sprites
- [X] Labels
- [X] Actions, Timer, Emitter
- [X] Music + Sounds

## TODO

- [ ] SpriteSheets + Animation
- [ ] Input handling
- [ ] Event broadcasting
- [ ] Framebuffers + Shader classes
- [ ] 3D render to texture (displayed in 2D)
- [ ] 2D Physics
- [ ] Bezier curves
- [ ] Video + GIFs
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