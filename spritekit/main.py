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

from . import _renderer as renderer
from .window import _init_window, get_window

__scene__ = []

def main(cls):
    global __scene__
    if __scene__:
        raise RuntimeError("@main already called")
    _init_window(*cls.window_size, cls.window_title, hints=cls.window_hints, frame_limit=cls.frame_limit)
    renderer.init()
    scn = cls()
    __scene__.append(scn)
    scn.enter()
    window = get_window()

    while not window.should_close:
        if not __scene__:
            window.quit()
        window.poll_events()
        scn.step(window.delta_time)
        scn.draw()
        renderer.flush()
        window.swap_buffers()

    return cls

def quit():
    window = get_window()
    window.should_close = True

__all__ = ['main', 'quit']