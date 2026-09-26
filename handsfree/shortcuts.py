from dataclasses import dataclass

from handsfree.config import ShortcutConfig


@dataclass
class ShortcutEvent:
    gesture: str
    binding: dict


class ShortcutEngine:
    """Fires a bound gesture once it has been held confidently for `hold_s`.

    Each hold fires at most once: the gesture must change (or the hand leave) before it can
    fire again, and a per-gesture cooldown guards against rapid re-triggers.
    """

    def __init__(self, cfg: ShortcutConfig):
        self.cfg = cfg
        self.candidate: str | None = None
        self._since = 0.0
        self._fired = False
        self._last_fire: dict[str, float] = {}

    def progress(self, t: float) -> float:
        """0..1 of the hold time elapsed for the current candidate (for the UI)."""
        if self.candidate is None or self._fired:
            return 0.0
        return min((t - self._since) / self.cfg.hold_s, 1.0) if self.cfg.hold_s > 0 else 1.0

    def update(self, label: str | None, confidence: float, t: float) -> list[ShortcutEvent]:
        bound = label in self.cfg.bindings and confidence >= self.cfg.min_confidence
        candidate = label if bound else None

        if candidate != self.candidate:
            self.candidate, self._since, self._fired = candidate, t, False
        if candidate is None or self._fired or t - self._since < self.cfg.hold_s:
            return []

        if t - self._last_fire.get(candidate, -float("inf")) < self.cfg.cooldown_s:
            return []  # Keep holding: fires as soon as the cooldown ends.
        self._fired = True
        self._last_fire[candidate] = t
        return [ShortcutEvent(candidate, self.cfg.bindings[candidate])]
