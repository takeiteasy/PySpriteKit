# spritekit/scene.py
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

from typing import Optional

from .actor import Parent
from .renderer import get_viewport, get_clear_color, set_viewport, set_clear_color 

import quickwindow

class Scene(Parent):
    config: dict = {}

    def __init__(self,
                 viewport: Optional[tuple[int, int]] = None,
                 clear_color: tuple[float, float, float, float] = (0, 0, 0, 1),
                 **kwargs):
        Parent.__init__(self)
        self.viewport = viewport if viewport is not None else quickwindow.size()
        self.clear_color = clear_color
    
    @property
    def viewport(self):
        return get_viewport()
    
    @viewport.setter
    def viewport(self, value: tuple[int, int]):
        set_viewport(value)
    
    @property
    def clear_color(self):
        return get_clear_color()
    
    @clear_color.setter
    def clear_color(self, value: tuple[float, float, float, float]):
        set_clear_color(value)
    
    def enter(self):
        pass

    def exit(self):
        pass

    def step(self, delta):
        for child in self.children:
            child.step(delta)

    def draw(self):
        for child in reversed(self.children):
            child.draw()