"""Deterministic peer registry and discovery for Agent 10."""

from __future__ import annotations

from collections.abc import Iterable

from .models import Capability, PeerProfile, PeerRole


class RegistryError(ValueError):
    """Base error for invalid registry operations."""


class DuplicatePeerError(RegistryError):
    """Raised when a peer ID is registered more than once."""


class UnknownPeerError(RegistryError):
    """Raised when an operation references an unregistered peer."""


class UnavailablePeerError(RegistryError):
    """Raised when a peer must be available for an operation."""


class IneligibleRoleError(RegistryError):
    """Raised when a peer claims a role outside its registered profile."""


class AdvertisementMismatchError(RegistryError):
    """Raised when a capability advertisement disagrees with the registry."""


class PeerRegistry:
    """Approved peer directory.

    The registry answers identity, capability, role-eligibility, and availability
    questions. It does not allocate work or choose which peer should act.
    """

    def __init__(self, profiles: Iterable[PeerProfile] = ()) -> None:
        self._profiles: dict[str, PeerProfile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: PeerProfile) -> None:
        if profile.agent_id in self._profiles:
            raise DuplicatePeerError(f"peer already registered: {profile.agent_id}")
        self._profiles[profile.agent_id] = profile

    def get(self, agent_id: str) -> PeerProfile:
        try:
            return self._profiles[agent_id]
        except KeyError as exc:
            raise UnknownPeerError(f"unknown peer: {agent_id}") from exc

    def all_profiles(self) -> tuple[PeerProfile, ...]:
        return tuple(self._profiles[agent_id] for agent_id in sorted(self._profiles))

    def set_availability(self, agent_id: str, available: bool) -> PeerProfile:
        profile = self.get(agent_id)
        updated = profile.model_copy(update={"available": available})
        self._profiles[agent_id] = updated
        return updated

    def require_available(self, agent_id: str) -> PeerProfile:
        profile = self.get(agent_id)
        if not profile.available:
            raise UnavailablePeerError(f"peer is unavailable: {agent_id}")
        return profile

    def require_role_eligibility(self, agent_id: str, role: PeerRole) -> PeerProfile:
        profile = self.require_available(agent_id)
        if role not in profile.eligible_roles:
            raise IneligibleRoleError(
                f"peer {agent_id} is not eligible for role {role.value}"
            )
        return profile

    def require_matching_advertisement(self, advertised: PeerProfile) -> PeerProfile:
        registered = self.require_available(advertised.agent_id)
        if advertised != registered:
            raise AdvertisementMismatchError(
                f"advertisement does not match registered profile: {advertised.agent_id}"
            )
        return registered

    def discover(
        self,
        *,
        capability: Capability | None = None,
        role: PeerRole | None = None,
        available_only: bool = True,
        exclude_agent_id: str | None = None,
    ) -> tuple[PeerProfile, ...]:
        """Return matching peers without assigning work to any of them."""

        matches: list[PeerProfile] = []
        for profile in self.all_profiles():
            if exclude_agent_id is not None and profile.agent_id == exclude_agent_id:
                continue
            if available_only and not profile.available:
                continue
            if capability is not None and capability not in profile.capabilities:
                continue
            if role is not None and role not in profile.eligible_roles:
                continue
            matches.append(profile)
        return tuple(matches)
