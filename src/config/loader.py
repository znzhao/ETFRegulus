"""Layered YAML config loading with strict typed validation.

Configs load into typed dataclasses, not raw dicts. An unknown key is an error, not a
silent no-op -- a typo in a hyperparameter name must not cost a training run
(reference/architecture.md section 3).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import types
import typing
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin

import yaml

T = TypeVar("T")

CONFIG_ROOT = Path("config")


class ConfigError(ValueError):
    """Raised for any malformed, unknown, or mistyped configuration key."""


# --------------------------------------------------------------------------- load


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge; `over` wins. Lists are replaced wholesale, not concatenated."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def resolve_config(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict:
    """Load a config, resolving `extends:` against config/ and the file's own directory.

    `extends` may be a string or a list. Later entries override earlier ones, and the
    file's own keys override everything it extends.
    """
    path = Path(path).resolve()
    if path in _seen:
        chain = " -> ".join(p.name for p in (*_seen, path))
        raise ConfigError(f"circular `extends` chain: {chain}")

    raw = load_yaml(path)
    parents = raw.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list):
        raise ConfigError(f"{path}: `extends` must be a string or list of strings")

    merged: dict = {}
    for parent in parents:
        candidates = [path.parent / parent, CONFIG_ROOT / parent, Path(parent)]
        for cand in candidates:
            if cand.exists():
                merged = _deep_merge(merged, resolve_config(cand, (*_seen, path)))
                break
        else:
            raise ConfigError(f"{path}: `extends` target not found: {parent}")

    return _deep_merge(merged, raw)


def config_hash(resolved: dict) -> str:
    """Stable hash of a resolved config. Goes into the run id and the manifest."""
    blob = json.dumps(resolved, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


# ----------------------------------------------------------------------- validate


def _is_optional(tp: Any) -> bool:
    return get_origin(tp) in (Union, types.UnionType) and type(None) in get_args(tp)


def _unwrap_optional(tp: Any) -> Any:
    args = [a for a in get_args(tp) if a is not type(None)]
    return args[0] if len(args) == 1 else Union[tuple(args)]


def _coerce(value: Any, tp: Any, path: str) -> Any:
    if tp is Any or tp is None:
        return value

    if _is_optional(tp):
        if value is None:
            return None
        tp = _unwrap_optional(tp)

    origin = get_origin(tp)

    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping for {tp.__name__}, got {type(value).__name__}")
        return strict_from_dict(tp, value, path)

    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        (item_tp,) = get_args(tp)[:1] or (Any,)
        items = [_coerce(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if origin is tuple else items

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        key_tp, val_tp = (get_args(tp) + (Any, Any))[:2]
        return {k: _coerce(v, val_tp, f"{path}.{k}") for k, v in value.items()}

    if origin is typing.Literal:
        if value not in get_args(tp):
            raise ConfigError(f"{path}: {value!r} is not one of {list(get_args(tp))}")
        return value

    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp is bool and not isinstance(value, bool):
        raise ConfigError(f"{path}: expected a bool, got {type(value).__name__}")
    if tp in (int, float, str) and not isinstance(value, tp):
        raise ConfigError(f"{path}: expected {tp.__name__}, got {type(value).__name__}")
    return value


def strict_from_dict(cls: type[T], data: dict, path: str = "") -> T:
    """Build a dataclass from a mapping. Unknown keys raise; missing required keys raise."""
    if not dataclasses.is_dataclass(cls):
        raise ConfigError(f"{cls!r} is not a dataclass")

    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}

    unknown = set(data) - set(fields)
    if unknown:
        near = ", ".join(sorted(fields))
        where = path or cls.__name__
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}. Known keys: {near}")

    kwargs: dict[str, Any] = {}
    for name, field in fields.items():
        sub = f"{path}.{name}" if path else name
        if name in data:
            kwargs[name] = _coerce(data[name], hints[name], sub)
        elif field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            continue
        else:
            raise ConfigError(f"{sub}: required key is missing")
    return cls(**kwargs)


def load_typed(path: str | Path, cls: type[T]) -> tuple[T, dict]:
    """Resolve `path` (following `extends`) and validate it into `cls`.

    Returns the typed config and the resolved raw dict (the dict is what gets hashed
    and snapshotted into the run manifest).
    """
    resolved = resolve_config(path)
    return strict_from_dict(cls, resolved), resolved
