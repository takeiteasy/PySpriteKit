# spritekit/state.py
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

import transitions
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

@dataclass
class Transition:
    trigger: str
    source: str | Enum | list
    dest: str | Enum
    conditions: Optional[str | list[str]] = None
    unless: Optional[str | list[str]] = None
    before: Optional[str | list[str]] = None
    after: Optional[str | list[str]] = None
    prepare: Optional[str | list[str]] = None
    kwargs: dict = field(default_factory=dict)

def _explode(t: Transition):
    a = {k: t.__dict__[k] for k in t.__annotations__.keys() if not k.startswith("_")}
    for k, v in a.items():
        if v is None:
            a.pop(k)
    b = a.pop("kwargs")
    a.update(**b)
    return a

class FiniteStateMachine:
    states: list[str | Enum] = []
    transitions: list[dict | Transition] = []

    def __init__(self, **kwargs):
        if self.states:
            if not "initial" in kwargs:
                kwargs["initial"] = self.states[0]
            self._machine = transitions.Machine(model=self,
                                                states=self.states,
                                                transitions=[_explode(t) if isinstance(t, Transition) else t for t in self.transitions],
                                                **kwargs)
        else:
            self._machine = None

__all__ = ["FiniteStateMachine", "Transition"]