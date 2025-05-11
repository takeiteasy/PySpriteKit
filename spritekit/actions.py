# spritekit/actions.py
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
from typing import Optional, Callable, Any, override
from contextlib import contextmanager
from queue import Queue

from .timer import TimerNode
from .actor import Actor
from .easing import ease_linear_in_out

class ActionType:
    pass

@dataclass
class ActionNode(ActionType, TimerNode):
    easing: Callable[[float, float, float, float], float] = ease_linear_in_out
    actor: Optional[Actor] = None
    field: Optional[str | list[str]] = None
    target: Any = None

    @contextmanager
    def _initial_value(self):
        obj = self.actor
        if self.field is None:
            return
        for i in range(len(self.field)):
            if i == len(self.field) - 1:
                f = getattr(obj, self.field[i])
                if f is None:
                    raise RuntimeError(f"Object has no field {self.field[i]}")
                if not isinstance(f, type(self.target)):
                    raise RuntimeError(f"Field {self.field[i]} is not of type {type(self.target)}")
                yield f
            else:
                if not hasattr(obj, self.field[i]):
                    raise RuntimeError(f"Object has no field {self.field[i]}")
                obj = getattr(obj, self.field[i])

    def __init__(self, **kwargs):
        new_kwargs = {
            "duration": kwargs.pop("duration", 1.),
            "cursor": kwargs.pop("cursor", None),
        }
        TimerNode.__init__(self, **new_kwargs)
        self.actor = kwargs.pop("actor", None)
        self.field = kwargs.pop("field", None)
        self.target = kwargs.pop("target", None)
        if self.actor is None:
            raise RuntimeError("Actor is not set")
        if self.field is None:
            raise RuntimeError("Field is not set")
        if self.target is None:
            raise RuntimeError("Target is not set")
        self.field = self.field if isinstance(self.field, list) else self.field.split(".") if "." in self.field else [self.field]
        self._initial_value()
        if "easing" in kwargs:
            self.easing = kwargs.pop("easing")
        self._start = None
        self.on_complete = self.remove_me
        self.on_tick = self._step

    @property
    def completed(self):
        return self._completed

    @property
    def running(self):
        return self._running

    def _step(self, _):
        if not self._start:
            with self._initial_value() as start:
                self._start = start
        elapsed = self.duration - self.position
        delta = self.target - self._start
        obj = self.actor
        for i in range(len(self.field)):
            if i == len(self.field) - 1:
                def fn(x, y, _z, w):
                    return self.easing(x, y, _z, w)
                f = getattr(obj, self.field[i])
                if isinstance(f, float) or isinstance(f, int):
                    v = type(f)(fn(elapsed, self._start, delta, self.duration))
                    setattr(obj, self.field[i], v)
                else:
                    z = list(zip(self._start, delta))
                    v = [fn(elapsed, start, delta, self.duration) for start, delta in z]
                    setattr(obj, self.field[i], v)
            else:
                obj = getattr(obj, self.field[i])

class WaitAction(ActionType, TimerNode):
    def __init__(self, **kwargs):
        self._on_complete_usr = kwargs.pop("on_complete", None)
        TimerNode.__init__(self, on_complete=self._on_complete, **kwargs)
        self._completed = False

    @property
    def completed(self):
        return self._completed

    def _on_complete(self):
        self._completed = True
        if self._on_complete_usr:
            self._on_complete_usr()
        self.remove_me()

class ActionSequence(ActionType, TimerNode, Queue):
    def __init__(self, actions: list[ActionType], duration: float = 0., auto_start: bool = True, remove_on_complete: bool = True, repeat: bool = False):
        Queue.__init__(self)
        self._actions = actions
        for action in actions:
            self.put(action)
        TimerNode.__init__(self,
                           duration=duration,
                           auto_start=auto_start,
                           remove_on_complete=remove_on_complete,
                           repeat=repeat)
        self._head = None
        if auto_start:
            self.start()

    def _complete(self):
        self._completed = True
        self._running = False
        if self.on_complete:
            self.on_complete()
        if self.repeat:
            self.reset()
            self.start()
        else:
            if self.remove_on_complete:
                self.remove_me()

    @override
    def step(self, delta: float):
        if not self._running or (self.empty() and not self._head):
            return
        if self._head is not None:
            self._head.step(delta)
            if self._head.completed:
                if self.empty():
                    self._complete()
                else:
                    self._head = None
        else:
            self._head = self.get()
            self._head.start()

    @override
    def reset(self):
        self._completed = False # noqa
        self._head = None
        self.queue.clear()
        for action in self._actions:
            self.put(action)

    @override
    def start(self):
        if self._running:
            return
        if self.empty():
            raise RuntimeError("ActionSequence is empty")
        self._running = True # noqa
        self._completed = False # noqa
        self._head = None

    @override
    def stop(self):
        self._complete()

__all__ = ["ActionNode", "WaitAction", "ActionSequence"]