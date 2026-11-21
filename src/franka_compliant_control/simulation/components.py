"""Scene extension contract. No controller, policy, task, Gym or network dependency."""

from __future__ import annotations
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET
import numpy as np

if TYPE_CHECKING:
    from .world import SimulationWorld
    from .contacts import ContactReport


class SceneComponent:
    """One component instance belongs to one world.

    build() may append MJCF before compilation. bind() resolves named IDs once.
    before_step() may change scene state/events; add_forces() MUST be pure except
    for adding to its output array (also called on an observer's shadow state).
    after_step() sees actual solver forces at EVERY physics substep.
    """

    def build(self, root: ET.Element, home: dict) -> None:
        pass

    def bind(self, world: SimulationWorld) -> None:
        if getattr(self, "_bound", False):
            raise ValueError("A scene component cannot be shared between worlds")
        self._bound = True  # no component -> world reference cycle

    def reset(self, world: SimulationWorld, rng: np.random.Generator) -> None:
        pass

    def before_step(self, world: SimulationWorld, dt: float) -> None:
        pass

    def add_forces(
        self, world: SimulationWorld, data, generalized_force: np.ndarray
    ) -> None:
        pass

    def after_step(
        self, world: SimulationWorld, contacts: ContactReport, dt: float
    ) -> None:
        pass

    def manifest(self) -> dict:
        return {"component": type(self).__name__}
