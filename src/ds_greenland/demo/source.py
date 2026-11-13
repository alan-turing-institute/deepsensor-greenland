"""Access to the Arraylake repo that backs the demo data dashboard."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Protocol

import zarr

if TYPE_CHECKING:
    import icechunk

DEFAULT_REPO = "aria-giant/hirham5-erai-modis-test"
DEFAULT_BRANCH = "main"


@dataclass(frozen=True)
class Snapshot:
    """Identity of one committed snapshot of the repo.

    Attributes
    ----------
    id : str
        Snapshot id.
    written_at : str
        ISO 8601 commit time.
    message : str
        Commit message.
    """

    id: str
    written_at: str
    message: str


class DataSource(Protocol):
    """Anything that can serve versioned zarr data to the dashboard."""

    repo: str
    branch: str
    root_path: str

    def tip_snapshot(self) -> Snapshot:
        """Return the snapshot currently at the tip of ``branch``."""
        ...

    def open_group(self, snapshot_id: str) -> zarr.Group:
        """Open the (read-only) group at ``root_path`` in ``snapshot_id``."""
        ...


class ArraylakeSource:
    """An Arraylake (Icechunk) repo, read via the ``arraylake`` SDK.

    Authentication uses ``token`` if given, otherwise the login cached by
    ``arraylake auth login``.

    Parameters
    ----------
    repo : str
        ``<org>/<repo>`` name on Arraylake.
    branch : str
        Branch to serve.
    root_path : str
        Group path within the repo holding the experiment groups; ``""`` is
        the repo root.
    token : str | None
        Arraylake API token.
    """

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        branch: str = DEFAULT_BRANCH,
        root_path: str = "",
        token: str | None = None,
    ) -> None:
        self.repo = repo
        self.branch = branch
        self.root_path = root_path
        self.token = token

    @classmethod
    def from_env(cls) -> ArraylakeSource:
        """Build from ``ARRAYLAKE_REPO``/``_BRANCH``/``_ROOT_PATH``/``_TOKEN``.

        Returns
        -------
        ArraylakeSource
            Source configured from the environment, falling back to defaults.
        """
        return cls(
            repo=os.environ.get("ARRAYLAKE_REPO", DEFAULT_REPO),
            branch=os.environ.get("ARRAYLAKE_BRANCH", DEFAULT_BRANCH),
            root_path=os.environ.get("ARRAYLAKE_ROOT_PATH", ""),
            token=os.environ.get("ARRAYLAKE_TOKEN"),
        )

    @cached_property
    def _repo(self) -> icechunk.Repository:
        try:
            from arraylake import Client
        except ImportError as e:
            raise ImportError(
                "The demo needs the `demo` dependency group on Python 3.12+; see "
                "the README's 'Demo site' section."
            ) from e

        return Client(token=self.token).get_repo(self.repo)

    def tip_snapshot(self) -> Snapshot:
        """Return the snapshot currently at the tip of ``branch``.

        This is a single metastore call, with no zarr reads.

        Returns
        -------
        Snapshot
            The tip-of-branch snapshot.
        """
        info = next(iter(self._repo.ancestry(branch=self.branch)))
        return Snapshot(
            id=info.id, written_at=info.written_at.isoformat(), message=info.message
        )

    def open_group(self, snapshot_id: str) -> zarr.Group:
        """Open the read-only group at ``root_path`` in ``snapshot_id``.

        Parameters
        ----------
        snapshot_id : str
            Snapshot to pin the read to.

        Returns
        -------
        zarr.Group
            The group at ``root_path``.
        """
        session = self._repo.readonly_session(snapshot_id=snapshot_id)
        group: Any = zarr.open_group(store=session.store, mode="r")
        return group[self.root_path] if self.root_path else group
