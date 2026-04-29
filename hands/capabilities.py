"""Registre de capacites locales readonly.

Iteration 5G-D : le registre expose une API de consultation au-dessus de
`LocalInventory`. Il permet a Jarvis de savoir si une cible locale existe, si
une action est disponible et quels garde-fous s'appliquent avant execution.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from hands.inventory import (
    CapabilityAction,
    CapabilityTargetType,
    LocalCapability,
    LocalInventory,
    LocalInventoryScanner,
)

_COMPACT_ALIASES = {
    "windowsdefender": ("windefend", "microsoftdefenderantivirusservice"),
    "microsoftdefender": ("windefend", "microsoftdefenderantivirusservice"),
    "defender": ("windefend", "microsoftdefenderantivirusservice"),
}


class InventoryScannerLike(Protocol):
    """Contrat minimal d'un scanner d'inventaire local."""

    def scan(self) -> LocalInventory:
        """Retourne un snapshot readonly de l'environnement local."""


class CapabilityDecision(BaseModel):
    """Decision de securite pour une action locale potentielle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    available: bool
    can_execute_now: bool
    capability: LocalCapability | None = None
    requires_confirmation: bool = False
    requires_admin: bool = False
    destructive: bool = False
    reason: str = Field(..., min_length=1)


class CapabilityRegistry:
    """Index consultable des capacites locales de Jarvis."""

    def __init__(
        self,
        *,
        scanner: InventoryScannerLike | None = None,
        inventory: LocalInventory | None = None,
    ) -> None:
        self._scanner = scanner or LocalInventoryScanner()
        self._inventory = inventory or self._scanner.scan()

    @property
    def inventory(self) -> LocalInventory:
        """Retourne le snapshot courant."""
        return self._inventory

    def refresh(self) -> LocalInventory:
        """Recharge le snapshot via le scanner configure."""
        self._inventory = self._scanner.scan()
        return self._inventory

    def find_capabilities(
        self,
        *,
        target_type: CapabilityTargetType | None = None,
        target_name: str | None = None,
        action: CapabilityAction | None = None,
    ) -> tuple[LocalCapability, ...]:
        """Filtre les capacites par type, cible et action."""
        matches = [
            capability
            for capability in self._inventory.capabilities
            if _matches_filter(
                capability,
                target_type=target_type,
                target_name=target_name,
                action=action,
            )
        ]
        return tuple(sorted(matches, key=_capability_sort_key))

    def can_execute(
        self,
        *,
        target_type: CapabilityTargetType,
        target_name: str,
        action: CapabilityAction,
    ) -> CapabilityDecision:
        """Indique si une action locale est connue et sous quelles conditions."""
        matches = self.find_capabilities(
            target_type=target_type,
            target_name=target_name,
            action=action,
        )
        if not matches:
            return CapabilityDecision(
                available=False,
                can_execute_now=False,
                reason="Capacite locale inconnue ou non detectee",
            )

        capability = _best_match(matches, target_name)
        can_execute_now = (
            capability.available
            and not capability.requires_confirmation
            and not capability.requires_admin
            and not capability.destructive
        )
        return CapabilityDecision(
            available=capability.available,
            can_execute_now=can_execute_now,
            capability=capability,
            requires_confirmation=capability.requires_confirmation,
            requires_admin=capability.requires_admin,
            destructive=capability.destructive,
            reason=_decision_reason(capability, can_execute_now),
        )


def _matches_filter(
    capability: LocalCapability,
    *,
    target_type: CapabilityTargetType | None,
    target_name: str | None,
    action: CapabilityAction | None,
) -> bool:
    if target_type is not None and capability.target_type != target_type:
        return False
    if action is not None and capability.action != action:
        return False
    return target_name is None or _name_matches(target_name, capability.target_name)


def _best_match(
    capabilities: Sequence[LocalCapability],
    target_name: str,
) -> LocalCapability:
    normalized_target = _normalize(target_name)
    compact_target = _compact(target_name)
    for capability in capabilities:
        if _normalize(capability.target_name) == normalized_target:
            return capability
    for capability in capabilities:
        if _compact(capability.target_name) == compact_target:
            return capability
    return capabilities[0]


def _decision_reason(capability: LocalCapability, can_execute_now: bool) -> str:
    if not capability.available:
        return f"Capacite detectee mais indisponible: {capability.reason}"
    if can_execute_now:
        return f"Action locale executable: {capability.reason}"
    blockers: list[str] = []
    if capability.requires_confirmation:
        blockers.append("confirmation requise")
    if capability.requires_admin:
        blockers.append("droits admin requis")
    if capability.destructive:
        blockers.append("action sensible")
    return f"Action locale encadree ({', '.join(blockers)}): {capability.reason}"


def _name_matches(query: str, candidate: str) -> bool:
    normalized_query = _normalize(query)
    normalized_candidate = _normalize(candidate)
    if normalized_query in normalized_candidate or normalized_candidate in normalized_query:
        return True

    query_compacts = _expanded_compacts(query)
    candidate_compacts = _expanded_compacts(candidate)
    return any(
        query_compact in candidate_compact or candidate_compact in query_compact
        for query_compact in query_compacts
        for candidate_compact in candidate_compacts
    )


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold().replace("\u2019", "'"))
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", stripped).strip()


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize(value))


def _expanded_compacts(value: str) -> frozenset[str]:
    compact = _compact(value)
    return frozenset((compact, *_COMPACT_ALIASES.get(compact, ())))


def _capability_sort_key(capability: LocalCapability) -> tuple[str, str, str]:
    return (capability.target_type, capability.target_name.casefold(), capability.action)
