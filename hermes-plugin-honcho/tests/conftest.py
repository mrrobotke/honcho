"""Load the plugin's session module without a Hermes install: stub the host modules it imports."""

from __future__ import annotations

import sys
import threading
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

PLUGIN_DIR = Path(__file__).resolve().parent.parent
PACKAGE = PLUGIN_DIR.name


def _spawn_context_thread(
    target: Callable[..., Any], *, name: str, daemon: bool = True, args: tuple = ()
) -> threading.Thread:
    return threading.Thread(target=target, name=name, daemon=daemon, args=args)


class _SingletonSlot:
    pass


def _stub(name: str, **attrs: Any) -> None:
    module = sys.modules.get(name) or types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


_stub("agent")
_stub("agent.memory_provider", spawn_context_thread=_spawn_context_thread)
_stub("agent.secret_scope", get_secret=lambda name, default=None: default)
_stub("agent.redact", redact_sensitive_text=lambda text, force=False: text)
_stub("hermes_cli")
_stub(
    "hermes_cli.profiles",
    _get_default_hermes_home=lambda: PLUGIN_DIR / ".hermes-test-home",
)
_stub("hermes_constants", get_hermes_home=lambda: PLUGIN_DIR / ".hermes-test-home")
_stub("hermes_state_common", TITLE_SOURCE_DERIVED="derived", TITLE_SOURCE_LLM="llm")
_stub("plugins")
_stub("plugins.plugin_utils", SingletonSlot=_SingletonSlot)

# A bare package over the plugin directory, so its submodules import without running the
# provider registration in __init__.py. The directory name is not an identifier, so pytest
# imports its __init__.py as a top-level "__init__" module; hand it the same bare package.
_package = types.ModuleType(PACKAGE)
_package.__file__ = str(PLUGIN_DIR / "__init__.py")
_package.__path__ = [str(PLUGIN_DIR)]
sys.modules[PACKAGE] = sys.modules["__init__"] = _package
