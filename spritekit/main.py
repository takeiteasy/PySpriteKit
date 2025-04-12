# spritekit/main.py
#
# Copyright (C) 2025 George Watson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from .scene import Scene

import quickwindow

__scene__ = []
__next_scene = None
__drop_scene = None

def main(cls):
    global __scene__, __drop_scene, __next_scene
    if __scene__:
        raise RuntimeError("@main already called")
    with quickwindow.quick_window(**cls.config) as wnd:
        scn = cls()
        __scene__.append(scn)
        scn.enter()
        
        for dt, events in wnd.loop():
            if not __scene__:
                break
            scn.step(dt)
            scn.draw()
            scn.renderer.flush()

            if __drop_scene:
                if isinstance(__drop_scene, list):
                    for _scn in reversed(__drop_scene):
                        _scn.exit()
                elif isinstance(__drop_scene, Scene):
                    __drop_scene.exit()
                else:
                    raise RuntimeError("Invalid Scene")
                __scene__ = __scene__[:-len(__drop_scene)]
                if __scene__:
                    scn = __scene__[-1]
                    scn.restored()
                __drop_scene = None
            if __next_scene:
                if isinstance(__next_scene, Scene):
                    if __scene__:
                        __scene__[-1].background()
                    __scene__.append(__next_scene)
                    scn = __next_scene
                    scn.enter()
                    __next_scene = None
                else:
                    raise RuntimeError("Invalid Scene")
    return cls