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
from .camera import Camera

class SceneType:
    pass

class SceneParent:
    def __init__(self, scenes: list[SceneType] | SceneType = []):
        self._scenes = scenes if isinstance(scenes, list) else [scenes]
        for scene in self._scenes:
            scene.parent = self
            scene.enter()
    
    def push_scene(self, scenes: SceneType | list[SceneType]):
        if not isinstance(scenes, list):
            scenes = [scenes]
        for scene in scenes:
            scene.parent = self
            scene.enter()
        self._scenes = scenes + self._scenes
    
    def pop_scene(self, n: int = 1):
        result = [self._scenes.pop(0) for _ in range(n)]
        for scene in result:
            scene.exit()
        return result[0] if n == 1 else result

    def append_scene(self, scenes: SceneType | list[SceneType]):
        if not isinstance(scenes, list):
            scenes = [scenes]
        for scene in scenes:
            scene.parent = self
            scene.enter()
        self._scenes.extend(scenes)

    def drop_scene(self, n: int = 1):
        result = [self._scenes.pop() for _ in range(n)]
        for scene in result:
            scene.exit()
        return result[0] if n == 1 else result
    
    def remove_scene(self, scene: SceneType):
        scene.exit()
        self._scenes.remove(scene)
    
    def clear_scenes(self):
        self._scenes = []
    
    def set_scene(self, scene: SceneType):
        scene.parent = self
        self._scenes = [scene]
    
    def step(self, delta):
        for scene in self._scenes:
            scene.step(delta)
    
    def draw(self):
        for scene in self._scenes:
            scene.draw()

class Scene(SceneType, ActorParent, FiniteStateMachine, SceneParent):
    background_color = (0., 0., 0., 1.)
    window_size = (640, 480)
    window_title = "SpriteKit"
    window_hints = None
    frame_limit = None

    def __init__(self, **kwargs):
        self.parent = None
        self._camera = kwargs.pop("camera", Camera())
        ActorParent.__init__(self)
        SceneParent.__init__(self, kwargs.pop("scenes", []))
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
    
    @property
    def camera(self):
        return self._camera
    
    @camera.setter
    def camera(self, value: Camera):
        self._camera = value
        self._camera.dirty = True
    
    def remove(self):
        if not self.parent:
            quit()
        else:
            self.parent.remove_scene(self)
    
    def enter(self):
        pass

    def exit(self):
        pass

    def step(self, delta):
        for child in self.children:
            child.step(delta)
        SceneParent.step(self, delta)

    def draw(self):
        if self.camera.dirty:
            renderer.set_world_matrix(self.camera.matrix)
        for child in reversed(self.children):
            child.draw()
        SceneParent.draw(self)

__all__ = ["Scene", "SceneParent", "SceneType"]