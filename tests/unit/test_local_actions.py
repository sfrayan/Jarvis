"""Tests unitaires du planificateur local."""

from __future__ import annotations

import pytest

from brain.events import IntentDomain, IntentRouted, IntentType
from config.schema import SafetyConfig, SafetyMode
from hands.capabilities import CapabilityDecision
from hands.executor import PlannedGuiAction
from hands.inventory import CapabilityAction, CapabilityTargetType, LocalCapability
from hands.local_actions import LocalActionPlanner

pytestmark = pytest.mark.unit


def _intent(
    text: str,
    *,
    domain: IntentDomain,
    intent: IntentType = "gui",
) -> IntentRouted:
    return IntentRouted(
        timestamp=1.0,
        original_text=text,
        normalized_text=text,
        intent=intent,
        domain=domain,
        confidence=0.9,
        reason="test",
        model="heuristic",
    )


class _FakeLocalActionBackend:
    def __init__(self, *, fail: bool = False) -> None:
        self.actions: list[PlannedGuiAction] = []
        self._fail = fail

    def perform(self, action: PlannedGuiAction) -> None:
        self.actions.append(action)
        if self._fail:
            raise RuntimeError("backend down")


class _FakeCapabilityRegistry:
    def __init__(
        self,
        *,
        unavailable: set[tuple[CapabilityTargetType, str, CapabilityAction]] | None = None,
        blocked: set[tuple[CapabilityTargetType, str, CapabilityAction]] | None = None,
        dynamic: tuple[LocalCapability, ...] = (),
    ) -> None:
        self.queries: list[tuple[CapabilityTargetType, str, CapabilityAction]] = []
        self.find_queries: list[
            tuple[CapabilityTargetType | None, str | None, CapabilityAction | None]
        ] = []
        self._unavailable = unavailable or set()
        self._blocked = blocked or set()
        self._dynamic = dynamic

    def find_capabilities(
        self,
        *,
        target_type: CapabilityTargetType | None = None,
        target_name: str | None = None,
        action: CapabilityAction | None = None,
    ) -> tuple[LocalCapability, ...]:
        self.find_queries.append((target_type, target_name, action))
        return tuple(
            capability
            for capability in self._dynamic
            if (target_type is None or capability.target_type == target_type)
            and (action is None or capability.action == action)
            and (
                target_name is None
                or target_name.casefold() in capability.target_name.casefold()
                or capability.target_name.casefold() in target_name.casefold()
            )
        )

    def can_execute(
        self,
        *,
        target_type: CapabilityTargetType,
        target_name: str,
        action: CapabilityAction,
    ) -> CapabilityDecision:
        key = (target_type, target_name, action)
        self.queries.append(key)
        if key in self._unavailable:
            return CapabilityDecision(
                available=False,
                can_execute_now=False,
                reason="Capacite locale inconnue ou non detectee",
            )

        is_blocked = key in self._blocked
        capability = LocalCapability(
            target_type=target_type,
            target_name=target_name,
            action=action,
            available=True,
            requires_confirmation=is_blocked,
            requires_admin=False,
            destructive=is_blocked,
            reason="test capability",
        )
        return CapabilityDecision(
            available=True,
            can_execute_now=not is_blocked,
            capability=capability,
            requires_confirmation=is_blocked,
            destructive=is_blocked,
            reason="Action locale encadree" if is_blocked else "Action locale executable",
        )


def _planner(
    mode: SafetyMode = "dry_run",
    *,
    backend: _FakeLocalActionBackend | None = None,
    capabilities: _FakeCapabilityRegistry | None = None,
) -> LocalActionPlanner:
    return LocalActionPlanner(
        SafetyConfig(mode=mode),
        backend=backend,
        capabilities=capabilities or _FakeCapabilityRegistry(),
    )


class TestLocalActionPlanner:
    @pytest.mark.parametrize(
        ("text", "domain", "action_type", "target"),
        [
            ("ouvre Spotify", "media", "launch_app", "Spotify"),
            ("ouvre Chrome", "apps", "launch_app", "Chrome"),
            ("ouvre Discord", "apps", "launch_app", "Discord"),
            ("ouvre les parametres", "apps", "launch_app", "Settings"),
            ("ouvre calculatrice", "apps", "launch_app", "Calculator"),
            ("ouvre bloc-notes", "apps", "launch_app", "Notepad"),
            ("ouvre paint", "apps", "launch_app", "Paint"),
            ("ouvre steam", "apps", "launch_app", "Steam"),
            ("ouvre origin", "apps", "launch_app", "Origin"),
            ("ouvre ea app", "apps", "launch_app", "EA App"),
            ("ouvre edge", "apps", "launch_app", "Edge"),
            ("ouvre firefox", "apps", "launch_app", "Firefox"),
            ("ouvre opera gx", "apps", "launch_app", "Opera GX"),
            ("ouvre le gestionnaire de tâches", "system", "system_tool", "Task Manager"),
            ("ouvre mes téléchargements", "folders", "open_folder", "Downloads"),
            ("volume monte", "system", "system_volume", "volume_up"),
            ("monte le volume", "system", "system_volume", "volume_up"),
            ("baisse le volume", "system", "system_volume", "volume_down"),
            ("coupe le volume", "system", "system_volume", "mute"),
            ("mets Spotify en pause", "media", "media_control", "pause"),
            ("arrete la musique", "media", "media_control", "pause"),
            ("stop la musique", "media", "media_control", "pause"),
            ("coupe la musique", "media", "media_control", "pause"),
            ("active la musique", "media", "media_control", "play"),
            ("mets de la musique", "media", "media_control", "play"),
        ],
    )
    def test_plans_supported_local_actions(
        self,
        text: str,
        domain: IntentDomain,
        action_type: str,
        target: str,
    ) -> None:
        planner = _planner()

        report = planner.plan(_intent(text, domain=domain))

        assert report is not None
        assert report.status == "dry_run"
        assert report.executed is False
        assert report.requires_human is False
        assert report.actions[0].type == action_type
        assert report.actions[0].text == target

    def test_shutdown_pc_is_blocked(self) -> None:
        backend = _FakeLocalActionBackend()
        planner = _planner("assisted", backend=backend)

        report = planner.plan(_intent("éteins le PC", domain="system"))

        assert report is not None
        assert report.status == "blocked"
        assert report.requires_human is True
        assert report.actions[0].destructive is True
        assert backend.actions == []

    @pytest.mark.parametrize("mode", ["observe", "dry_run"])
    def test_non_execution_modes_do_not_call_backend(self, mode: SafetyMode) -> None:
        backend = _FakeLocalActionBackend()
        planner = _planner(mode, backend=backend)

        report = planner.plan(_intent("ouvre Chrome", domain="apps"))

        assert report is not None
        assert report.executed is False
        if mode == "observe":
            assert report.status == "observe"
        else:
            assert report.status == "dry_run"
        assert report.requires_human is False
        assert backend.actions == []

    @pytest.mark.parametrize("mode", ["assisted", "autonomous"])
    def test_execution_modes_call_backend_for_safe_actions(self, mode: SafetyMode) -> None:
        backend = _FakeLocalActionBackend()
        planner = _planner(mode, backend=backend)

        report = planner.plan(_intent("ouvre Chrome", domain="apps"))

        assert report is not None
        assert report.status == "completed"
        assert report.executed is True
        assert report.requires_human is False
        assert len(backend.actions) == 1
        assert backend.actions[0].type == "launch_app"
        assert backend.actions[0].text == "Chrome"

    def test_backend_failure_blocks_action(self) -> None:
        backend = _FakeLocalActionBackend(fail=True)
        planner = _planner("assisted", backend=backend)

        report = planner.plan(_intent("ouvre Chrome", domain="apps"))

        assert report is not None
        assert report.status == "blocked"
        assert report.executed is False
        assert report.requires_human is True
        assert "RuntimeError" in report.reason
        assert len(backend.actions) == 1

    def test_unsupported_domain_returns_none(self) -> None:
        planner = _planner()

        report = planner.plan(_intent("ouvre Gmail", domain="google_workspace"))

        assert report is None

    def test_chat_intent_returns_none(self) -> None:
        planner = _planner()

        report = planner.plan(_intent("ouvre Chrome", domain="apps", intent="chat"))

        assert report is None

    def test_unknown_local_command_returns_none(self) -> None:
        planner = _planner()

        report = planner.plan(_intent("fais un truc local", domain="system"))

        assert report is None

    def test_youtube_browser_request_does_not_open_spotify(self) -> None:
        planner = _planner()

        report = planner.plan(
            _intent(
                "Chrome ouvre un nouvel onglet et fait une recherche sur YouTube",
                domain="media",
            )
        )

        assert report is None

    def test_queries_registry_before_planning_supported_action(self) -> None:
        capabilities = _FakeCapabilityRegistry()
        planner = _planner(capabilities=capabilities)

        report = planner.plan(_intent("ouvre Chrome", domain="apps"))

        assert report is not None
        assert capabilities.queries == [("app", "Chrome", "open")]

    def test_registry_unknown_blocks_known_alias(self) -> None:
        capabilities = _FakeCapabilityRegistry(unavailable={("app", "Chrome", "open")})
        planner = _planner(capabilities=capabilities)

        report = planner.plan(_intent("ouvre Chrome", domain="apps"))

        assert report is not None
        assert report.status == "blocked"
        assert report.executed is False
        assert report.requires_human is True
        assert "inconnue" in report.reason

    def test_dynamic_app_from_registry_is_planned(self) -> None:
        capabilities = _FakeCapabilityRegistry(
            dynamic=(
                LocalCapability(
                    target_type="app",
                    target_name="Obsidian",
                    action="open",
                    available=True,
                    requires_confirmation=False,
                    requires_admin=False,
                    destructive=False,
                    reason="detected app",
                ),
            )
        )
        planner = _planner(capabilities=capabilities)

        report = planner.plan(_intent("ouvre Obsidian", domain="apps"))

        assert report is not None
        assert report.status == "dry_run"
        assert report.actions[0].type == "launch_app"
        assert report.actions[0].text == "Obsidian"
        assert capabilities.find_queries == [("app", "obsidian", "open")]
        assert capabilities.queries == [("app", "Obsidian", "open")]

    def test_close_app_is_blocked_by_registry_confirmation(self) -> None:
        capabilities = _FakeCapabilityRegistry(blocked={("app", "Discord", "close")})
        planner = _planner(capabilities=capabilities)

        report = planner.plan(_intent("ferme Discord", domain="apps"))

        assert report is not None
        assert report.status == "blocked"
        assert report.executed is False
        assert report.requires_human is True
        assert report.actions[0].destructive is True
