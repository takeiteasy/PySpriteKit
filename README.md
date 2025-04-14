# PySpriteKit

> [!WARNING]
> Work in progress, see [TODO](#todo)

> [!NOTE]
> `pip install spritekit==0.2.1`

2D scene+actor framework based off Apple's SpriteKit. See [test.py](https://github.com/takeiteasy/PySpriteKit/blob/master/test.py) for working example.

## Features

- [X] Easy + Flexible
- [X] Scenes
- [X] Automatic batched rendering
- [X] Automatic asset cache
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
- [ ] Render scenes to Framebuffers
- [ ] 3D render to texture (displayed in 2D)
- [ ] 2D Physics
- [ ] Bezier curves
- [ ] Video + GIFs
- [ ] Headless mode
- [ ] Add examples
- [ ] Add documentation

## Requirements

```
transitions==0.9.2
pyglm==2.8.1
raudio==0.0.1
moderngl==5.12.0
numpy==2.2.4
pillow==11.1.0
pyglsl==0.0.8
```

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