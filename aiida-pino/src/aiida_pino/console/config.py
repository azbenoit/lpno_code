from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Console backend settings. Mirrors the web demo backend's Settings contract
    (env-prefixed, one worker, CORS pinned to one page origin) but for the
    aiida-pino workflow console instead of an inference server."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AIIDA_PINO_WEB_", case_sensitive=False)

    app_name: str = "aiida-pino-console"
    host: str = "0.0.0.0"
    port: int = 8100
    uvicorn_workers: int = 1
    log_level: str = "INFO"

    # Load the AiiDA profile at startup. Off => routes are live but the service
    # is None (returns 503), useful for contract tests without a profile.
    startup_load_service: bool = True
    # Empty string loads the default profile; set to pick a named one.
    aiida_profile: str = ""

    # Browser page origin the backend trusts for cross-origin dev (run_local.sh
    # sets AIIDA_PINO_WEB_PAGES_ORIGIN=http://<ip>:<frontend-port>). Empty in
    # production behind a same-origin proxy.
    pages_origin: str = ""

    #: flow3D_scripts checkout the console's FLOW-3D submits pin. Empty means
    #: "let the adapter resolve it", which falls through to the bundled
    #: `external/flow3D_scripts` submodule — pinned at 2026-07-01, a month
    #: before the calibration campaign, so it silently ignores `recoil_model`
    #: and `void_vapor_pressure_pa`. Every other launch path pins a root; the
    #: console was the only one that did not (3 of 1178 runs, all console).
    #: Set AIIDA_PINO_WEB_FLOW3D_SCRIPTS_ROOT to the current checkout.
    flow3d_scripts_root: str = ""

    # List/pagination caps for the browse endpoints.
    list_default_limit: int = 100
    list_max_limit: int = 500
    tail_max_lines: int = 1000
    logs_max_chars: int = 200_000

    # POST body limits. Submit bodies carry profile dicts and, for FLOW-3D
    # generate, a full verbatim run_config JSON, so allow more than reads.
    request_max_bytes: int = 64 * 1024
    submit_max_bytes: int = 2 * 1024 * 1024

    @field_validator("uvicorn_workers")
    @classmethod
    def _validate_workers(cls, value: int) -> int:
        if value != 1:
            # One worker keeps a single loaded AiiDA profile per process and one
            # shared view of the daemon/DB; multiple workers gain nothing here.
            raise ValueError("uvicorn_workers must be 1")
        return value

    @property
    def cors_allow_origins(self) -> list[str]:
        """Every origin the console page is actually served from.

        `pages_origin` accepts a comma-separated list. In LAN mode the page is
        pinned to the workstation's IP, so browsing the README's own quick-start
        URL (http://127.0.0.1:8180/index.html) served a page whose every fetch
        was blocked by CORS -- reported to the user as "Failed to fetch", with
        no mention of CORS, the config file, or the URL that works. The
        loopback aliases of the same port are added for that reason.
        """
        origins: list[str] = []
        for raw in self.pages_origin.split(","):
            origin = raw.strip().rstrip("/")
            if origin and origin not in origins:
                origins.append(origin)
        for origin in list(origins):
            for host in ("127.0.0.1", "localhost"):
                scheme, _, rest = origin.partition("://")
                port = rest.partition(":")[2]
                alias = f"{scheme}://{host}:{port}" if port else f"{scheme}://{host}"
                if alias not in origins:
                    origins.append(alias)
        return origins

    def clamp_limit(self, limit: int | None) -> int | None:
        """Clamp a requested page size. A request for ``0`` means *no cap*, and
        is returned as **None**, not as 0.

        A browsable store needs an escape hatch from paging: with 538 active
        collections even the 500 maximum truncated, so "list everything" had no
        expressible form.

        The sentinel is None because that is the value which is already correct
        everywhere it lands — `nodes[:None]` is the whole list,
        `QueryBuilder.limit(None)` is unlimited, and `if limit` is False. The
        first version returned 0 and required every call site to remember a
        guard; two of the four forgot, so "Show all" silently answered **zero**
        models, zero processes and zero simulations while working for datasets.
        A sentinel that has to be remembered is a defect waiting for its next
        call site.
        """
        if limit is None:
            return self.list_default_limit
        limit = int(limit)
        if limit < 0:
            # Only 0 is the sentinel. `<= 0` let -1 inherit it, so a `parseInt`
            # over an empty input -- which yields NaN, or -1 from a decrementing
            # control -- silently fired the most expensive query in the store.
            from .errors import InvalidRequestError

            raise InvalidRequestError(f"limit must be 0 (no cap) or a positive number, got {limit}")
        if limit == 0:
            return None
        return max(1, min(self.list_max_limit, limit))
