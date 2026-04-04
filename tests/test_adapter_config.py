import json

import pytest

from turborag.adapters.config import (
    ADAPTER_CONFIG_FILE_NAME,
    build_fetch_records_from_config,
    default_adapter_config_path,
    load_adapter_config,
    maybe_load_adapter_config,
    normalize_adapter_backend,
    save_adapter_config,
    validate_adapter_config,
)


def test_default_adapter_config_path_uses_index_directory(tmp_path):
    target = default_adapter_config_path(tmp_path / "index")
    assert target.name == ADAPTER_CONFIG_FILE_NAME
    assert target.parent == tmp_path / "index"


def test_save_and_load_adapter_config_round_trip(tmp_path):
    config = {
        "schema_version": 1,
        "backend": "neon",
        "options": {"dsn": "${DATABASE_URL}", "table": "public.chunks"},
    }
    path = tmp_path / "index" / ADAPTER_CONFIG_FILE_NAME
    save_adapter_config(config, path)

    loaded = load_adapter_config(path)
    assert loaded["backend"] == "neon"
    assert loaded["options"]["dsn"] == "${DATABASE_URL}"


def test_maybe_load_adapter_config_detects_default_file(tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir(parents=True, exist_ok=True)
    config_path = index_path / ADAPTER_CONFIG_FILE_NAME
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "neon",
                "options": {"dsn": "postgresql://example"},
            }
        ),
        encoding="utf-8",
    )

    loaded, path = maybe_load_adapter_config(index_path)
    assert loaded is not None
    assert path == config_path


def test_normalize_adapter_backend_aliases():
    assert normalize_adapter_backend("postgresql") == "postgres"
    assert normalize_adapter_backend("supabase-postgres") == "supabase_postgres"


def test_validate_adapter_config_for_neon_requires_dsn():
    with pytest.raises(ValueError):
        validate_adapter_config({"backend": "neon", "options": {}})

    validated = validate_adapter_config(
        {
            "backend": "neon",
            "options": {"dsn": "${DATABASE_URL}", "table": "public.chunks"},
        },
        resolve_env=False,
    )
    assert validated["backend"] == "neon"


def test_build_fetch_records_from_config_unsupported_backend_errors():
    with pytest.raises(ValueError):
        build_fetch_records_from_config(
            {
                "backend": "unknown",
                "options": {},
            }
        )
