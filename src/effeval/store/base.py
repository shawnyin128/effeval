"""What planning needs from a result store, independent of SQLite or HTTP."""

from __future__ import annotations

from typing import Protocol

from effeval.core.types import Artifact, Record


class ReuseLookup(Protocol):
    def find_reusable(
        self, run_key: str, *, viewer: str | None, need: frozenset[Artifact], hardware: str | None
    ) -> Record | None:
        """Newest record with this RunKey that has every artifact in `need`.

        Only records visible to `viewer` (own or public), never probe or another owner's retracted record, and
        marked reusable count;
        restricted to `hardware` when given.
        """
        ...
