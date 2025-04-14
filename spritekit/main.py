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

__scene__ = None
__next_scene__ = None

def _load_scene(cls):
    global __scene__
    if __scene__ is not None:
        __scene__.exit()
    __scene__ = cls()
    __scene__.enter()

def load_scene(cls):
    global __next_scene__
    __next_scene__ = cls

def main(cls):
    _init_window(*cls.window_size, cls.window_title, hints=cls.window_hints, frame_limit=cls.frame_limit)
    renderer.init()
    load_scene(cls)
    window = get_window()
    while not window.should_close:
        window.poll_events()
        __scene__.step(window.delta_time)
        __scene__.draw()
        renderer.flush()
        window.swap_buffers()
        if __next_scene__ is not None:
            _load_scene(__next_scene__)
            __next_scene__ = None
    return cls

def quit():
    window = get_window()
    window.should_close = True

__all__ = ['main', 'quit', 'load_scene']