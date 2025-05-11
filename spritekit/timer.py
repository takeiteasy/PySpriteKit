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
from typing import Optional, Callable

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
        self._completed = False
        self._running = self.auto_start
        if self.repeat is None:
            self.repeat = False
        elif isinstance(self.repeat, bool):
            pass
        elif isinstance(self.repeat, int):
            if self.repeat <= 0:
                self.repeat = False
        self._initial_repeat = self.repeat
        self._position = self.duration if self._running else 0
        if self.remove_on_complete is None:
            if isinstance(self.repeat, bool):
                self.remove_on_complete = not self.repeat
            elif isinstance(self.repeat, int):
                self.remove_on_complete = self.repeat > 0
            else:
                self.remove_on_complete = False

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value: float):
        self._position = max(min(value, self.duration), 0)  # noqa
        if self.on_tick:
            self.on_tick(self._position)

    def step(self, delta: float):
        if not self._running or self._completed:
            return
        if self._position is not None and self._position > 0:
            self._position -= delta
            if self._position <= 0:
                self._position = 0
                self._running = False
                self._completed = True
                if self.on_complete:
                    self.on_complete()
                if self.remove_on_complete:
                    self.remove_me()
                if self.repeat is not None:
                    self._position = self.duration
                    if isinstance(self.repeat, bool):
                        if self.repeat:
                            self.reset()
                    elif isinstance(self.repeat, int):
                        if self.repeat > 0:
                            self.repeat -= 1
                            self.reset()
            else:
                if self.on_tick:
                    self.on_tick(self._position)

    def reset(self):
        self._completed = False
        self.repeat = self._initial_repeat
        self._position = self.duration
        if self.auto_start:
            self.start()

    def start(self):
        if not self._running:
            self._running = True
            self._completed = False
            self._position = self.duration

    def stop(self):
        self._running = False
        self._completed = True
        self._position = 0

    def pause(self):
        if not self._completed:
            self._running = False

    def resume(self):
        if not self._completed:
            self._running = True

__all__ = ["TimerNode"]