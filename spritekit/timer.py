# spritekit/timer.py
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

from dataclasses import dataclass
from typing import Optional, Callable, Type

from .actor import Actor

@dataclass
class TimerType:
    duration: float = 1.
    repeat: Optional[bool | int] = None
    on_complete: Optional[Callable[[], None]] = None
    on_tick: Optional[Callable[[float], None]] = None
    auto_start: bool = True
    remove_on_complete: Optional[bool] = None

class TimerNode(TimerType, Actor):
    def __init__(self, **kwargs):
        Actor.__init__(self,
                       name=kwargs.pop("name", None),
                       on_added=kwargs.pop("on_added", None),
                       on_removed=kwargs.pop("on_removed", None))
        TimerType.__init__(self, **kwargs)
        self._position = self.duration
        self._running = self.auto_start

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value: float):
        self._position = max(min(value, self.duration), 0)  # noqa
        if self.on_tick:
            self.on_tick(self._position)

    def _end(self):
        self._running = False
        self._position = 0
        if self.remove_on_complete:
            self.remove_me()

    def step(self, delta: float):
        if not self._running:
            return
        self._position -= delta
        if self.on_tick:
            self.on_tick(self._position)
        if self._position <= 0:
            if self.on_complete:
                self.on_complete()
            match self.repeat:
                case bool():
                    if self.repeat:
                        self._position = self.duration
                    else:
                        self._end()
                case int():
                    self.repeat -= 1
                    if self.repeat <= 0:
                        self._end()
                case _:
                    self._end()

    def start(self):
        self._running = True
        self._position = self.duration

    def stop(self):
        self._running = False
        self._position = self.duration

    def pause(self):
        self._running = False

    def resume(self):
        self._running = True


class EmitterNode(TimerNode):
    def __init__(self,
                 repeat: bool | int = True,
                 emit: Callable[[], Actor] | Actor | tuple[Type[Actor], dict] = None,
                 **kwargs):
        self._user_on_complete = kwargs.pop("on_complete", None)
        if callable(emit):
            self._emit = staticmethod(emit)
        elif isinstance(emit, Actor):
            self._copy = emit.clone()
            self._emit = staticmethod(self._clone)
        else:
            self._type = emit[0]
            self._args = emit[1]
            self._emit = staticmethod(self._create)
        super().__init__(on_complete=self._spawn, repeat=repeat, **kwargs)
        if self.auto_start:
            self._emit()

    def _clone(self):
        return self._copy.clone()

    def _create(self):
        return self._type(**self._args)

    def _spawn(self):
        assert self.parent is not None, "Emitter must have a parent"
        assert callable(self._emit), "Emitter must be callable"
        self.parent.add(self._emit())
        if self._user_on_complete:
            self._user_on_complete()

__all__ = ["TimerNode", "EmitterNode"]