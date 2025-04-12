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

from .actor import Actor
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class Timer(Actor):
    duration: float = 1.
    repeat: Optional[bool | int] = None
    on_complete: Optional[Callable[[], None]] = None
    on_tick: Optional[Callable[[float], None]] = None
    auto_start: bool = True
    remove_on_complete: Optional[bool] = None
    cursor: Optional[float] = None

    def __post_init__(self):
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
        if self.cursor is None:
            self.cursor = self.duration if self._running else 0
        if self.remove_on_complete is None:
            if isinstance(self.repeat, bool):
                self.remove_on_complete = not self.repeat
            elif isinstance(self.repeat, int):
                self.remove_on_complete = self.repeat > 0
            else:
                self.remove_on_complete = False

    def step(self, delta: float):
        if not self._running or self._completed:
            return
        if self.cursor is not None and self.cursor > 0:
            self.cursor -= delta
            if self.cursor <= 0:
                self.cursor = 0
                self._running = False
                self._completed = True
                if self.on_complete:
                    self.on_complete()
                if self.remove_on_complete:
                    self.remove()
                if self.repeat is not None:
                    self.cursor = self.duration
                    if isinstance(self.repeat, bool):
                        if self.repeat:
                            self.reset()
                    elif isinstance(self.repeat, int):
                        if self.repeat > 0:
                            self.repeat -= 1
                            self.reset()
            else:
                if self.on_tick:
                    self.on_tick(self.cursor)

    def reset(self):
        self._completed = False
        self.repeat = self._initial_repeat
        self.cursor = self.duration
        if self.auto_start:
            self.start()

    def start(self):
        if not self._running:
            self._running = True
            self._completed = False
            self.cursor = self.duration

    def stop(self):
        self._running = False
        self._completed = True
        self.cursor = 0

    def pause(self):
        if not self._completed:
            self._running = False

    def resume(self):
        if not self._completed:
            self._running = True

__all__ = ["Timer"]