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

from .actor import ActorParent
from . import _renderer as renderer
from . import _drawable as drawable
from .state import FiniteStateMachine
from .window import window_size
from .main import quit

class Scene(ActorParent, FiniteStateMachine):
    background_color = (0., 0., 0., 1.)
    window_size = (640, 480)
    window_title = "SpriteKit"
    window_hints = None
    frame_limit = None

    def __init__(self, **kwargs):
        ActorParent.__init__(self)
        FiniteStateMachine.__init__(self, **kwargs)
        self.viewport = window_size()
        self.clear_color = self.__class__.background_color
    
    @property
    def viewport(self):
        return renderer.get_viewport()
    
    @viewport.setter
    def viewport(self, value: tuple[int, int]):
        renderer.set_viewport(value)
    
    @property
    def clear_color(self):
        return renderer.get_clear_color()
    
    @clear_color.setter
    def clear_color(self, value: tuple | list):
        renderer.set_clear_color(drawable.convert_color(value))
    
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

class Director(Scene):
    def __init__(self, initial_scene: Scene, **kwargs):
        super().__init__(**kwargs)
        self._next_scene = None
        self._scene = initial_scene
        self._scene.enter()
    
    @property
    def scene(self):
        return self._scene
    
    @scene.setter
    def scene(self, value: Scene):
        self._next_scene = value
    
    def step(self, delta):
        if not self._scene:
            quit()
        if self._next_scene:
            if self._scene:
                self._scene.exit()
            self._scene = self._next_scene
            self._scene.enter()
            self._next_scene = None
        self._scene.step(delta)

    def draw(self):
        self._scene.draw()

__all__ = ["Scene", "Director"]