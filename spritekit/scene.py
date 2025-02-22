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

from .actor import ActorType, ActorParent
from .fsm import FiniteStateMachine
import pyray as r
from typing import override

__all__ = ["Scene", "main_scene"]

__scene__ = []
__next_scene = None
__drop_scene = None

class Scene(FiniteStateMachine, ActorParent):
    config: dict = {}

    def __init__(self, **kwargs):
        FiniteStateMachine.__init__(self, **kwargs)
        self.camera = r.Camera2D()
        self.camera.target = 0, 0
        self.camera.offset = r.get_screen_width() / 2, r.get_screen_height() / 2
        self.camera.zoom = 1.
        self.clear_color = r.RAYWHITE
        self.run_in_background = False
        self.assets = {} # TODO: Store and restore assets to __cache in raylib.py
    
    @override
    def add_child(self, node: ActorType):
        if node:
            node.__dict__["scene"] = self
            self._add_child(node)
        else:
            raise RuntimeError("Invalid Node")

    def enter(self):
        pass

    def reenter(self):
        pass

    def background(self):
        pass

    def exit(self):
        pass

    def step(self, delta):
        for child in self.all_children():
            child.step(delta)

    def step_background(self, delta):
        if self.run_in_background:
            self.step(delta)

    def draw(self):
        r.clear_background(self.clear_color)
        r.begin_mode_2d(self.camera)
        for child in self.all_children():
            child.draw()
        r.end_mode_2d()

    def draw_background(self):
        if self.run_in_background:
            self.draw()
    
    @classmethod
    def push_scene(cls, scene):
        if not isinstance(scene, Scene):
            raise RuntimeError("Invalid Scene")
        global __next_scene
        if __next_scene is not None:
            raise RuntimeError("Next scene already queued")
        __next_scene = scene

    @classmethod
    def drop_scene(cls):
        global __scene__, __drop_scene
        if __drop_scene is not None:
            raise RuntimeError("Drop scene already queued")
        __drop_scene = __scene__[-1:]

    @classmethod
    def first_scene(cls):
        global __scene__, __drop_scene
        __drop_scene = __scene__[1:]

    @classmethod
    def current_scene(cls):
        if not __scene__:
            raise RuntimeError("No active Scene")
        return __scene__[-1]
    
    @classmethod
    def main_scene(cls):
        global __next_scene
        if __next_scene is None:
            raise RuntimeError("No next scene queued")
        return __scene__[0]

def main_scene(cls):
    global __scene__, __drop_scene, __next_scene
    if __scene__:
        raise RuntimeError("There can only be one @main_scene")
    r.init_window(cls.config['width'] if "width" in cls.config else 800,
                  cls.config['height'] if "height" in cls.config else 600,
                  cls.config['title'] if "title" in cls.config else "spritekit")
    r.set_config_flags(cls.config['flags'] if "flags" in cls.config else r.ConfigFlags.FLAG_WINDOW_RESIZABLE)
    r.init_audio_device()
    if "fps" in cls.config:
        r.set_target_fps(cls.config['fps'])
    if "exit_key" in cls.config:
        r.set_exit_key(cls.config['exit_key'])
    scn = cls()
    __scene__.append(scn)
    scn.enter()
    while not r.window_should_close() and __scene__:
        dt = r.get_frame_time()
        if len(__scene__) > 1:
            for _scn in __scene__[:-1]:
                _scn.step_background(dt)
        scn.step(dt)
        r.begin_drawing()
        if len(__scene__) > 1:
            for _scn in __scene__[:-1]:
                _scn.draw_background()
        scn.draw()
        r.end_drawing()
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
                scn.reenter()
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