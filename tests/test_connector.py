"""Unit tests for the Snowflake connector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import snowflake.sqlalchemy.custom_types as sct
from sqlalchemy import types

import target_snowflake.connector as connector_module
from target_snowflake.connector import SnowflakeConnector, SnowflakeTimestampType
from target_snowflake.snowflake_types import NUMBER, VARIANT


@pytest.fixture
def connector():
    return SnowflakeConnector()


@pytest.mark.parametrize(
    ("schema", "expected_type"),
    [
        pytest.param({"type": "object"}, VARIANT, id="object"),
        pytest.param({"type": ["array", "null"]}, VARIANT, id="array"),
        pytest.param({"type": ["array", "object", "string"]}, VARIANT, id="array_object_string"),
        pytest.param({"type": ["integer", "null"]}, NUMBER, id="integer"),
        pytest.param({"type": ["number", "null"]}, sct.DOUBLE, id="number"),
        pytest.param({"type": ["string", "null"], "format": "date-time"}, sct.TIMESTAMP_NTZ, id="date-time"),
        # Upstream types
        pytest.param({"type": ["string", "null"]}, types.VARCHAR, id="string"),
        pytest.param({"type": ["boolean", "null"]}, types.BOOLEAN, id="boolean"),
        pytest.param({"type": "string", "format": "time"}, types.TIME, id="time"),
        pytest.param({"type": "string", "format": "date"}, types.DATE, id="date"),
        pytest.param({"type": "string", "format": "uuid"}, types.UUID, id="uuid"),
    ],
)
def test_jsonschema_to_sql(connector: SnowflakeConnector, schema: dict, expected_type: type[types.TypeEngine]):
    sql_type = connector.to_sql_type(schema)
    assert isinstance(sql_type, expected_type)


@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        ({"timestamp_type": SnowflakeTimestampType.TIMESTAMP_TZ}, sct.TIMESTAMP_TZ),
        ({"timestamp_type": SnowflakeTimestampType.TIMESTAMP_LTZ}, sct.TIMESTAMP_LTZ),
        ({"timestamp_type": SnowflakeTimestampType.TIMESTAMP_NTZ}, sct.TIMESTAMP_NTZ),
    ],
)
def test_datetime_to_sql(connector: SnowflakeConnector, config: dict, expected_type: type[types.TypeEngine]):
    connector.config.update(config)
    schema = {"type": ["string", "null"], "format": "date-time"}
    sql_type = connector.to_sql_type(schema)
    assert isinstance(sql_type, expected_type)


def test_to_sql_type_with_max_varchar_length(connector: SnowflakeConnector):
    sql_type = connector.to_sql_type({"type": "string", "maxLength": 1_000_000})
    assert isinstance(sql_type, types.VARCHAR)
    assert sql_type.length == 1_000_000

    sql_type = connector.to_sql_type({"type": "string", "maxLength": SnowflakeConnector.max_varchar_length + 1})
    assert isinstance(sql_type, types.VARCHAR)
    assert sql_type.length == SnowflakeConnector.max_varchar_length


def test_email_format(connector: SnowflakeConnector):
    sql_type = connector.to_sql_type({"type": "string", "format": "email"})
    assert isinstance(sql_type, types.VARCHAR)
    assert sql_type.length == 254


def test_uri_format(connector: SnowflakeConnector):
    sql_type = connector.to_sql_type({"type": "string", "format": "uri"})
    assert isinstance(sql_type, types.VARCHAR)
    assert sql_type.length == 2083


def test_hostname_format(connector: SnowflakeConnector):
    sql_type = connector.to_sql_type({"type": "string", "format": "hostname"})
    assert isinstance(sql_type, types.VARCHAR)
    assert sql_type.length == 253


def test_ipv4_format(connector: SnowflakeConnector):
    sql_type = connector.to_sql_type({"type": "string", "format": "ipv4"})
    assert isinstance(sql_type, types.VARCHAR)
    assert sql_type.length == 15


def test_ipv6_format(connector: SnowflakeConnector):
    sql_type = connector.to_sql_type({"type": "string", "format": "ipv6"})
    assert isinstance(sql_type, types.VARCHAR)
    assert sql_type.length == 45


def test_singer_decimal(connector: SnowflakeConnector):
    sql_type = connector.to_sql_type(
        {
            "type": "string",
            "format": "x-singer.decimal",
            "precision": 38,
            "scale": 18,
        },
    )
    assert isinstance(sql_type, types.DECIMAL)
    assert sql_type.precision == 38
    assert sql_type.scale == 18


def _oauth_connector(database: str = "CUSTOMER_DB") -> SnowflakeConnector:
    """OAuth-configured connector."""
    return SnowflakeConnector(
        config={
            "account": "lzc69188.us-east-1",
            "user": "u",
            "database": database,
            "warehouse": "COMPUTE_WH",
            "role": "PUBLIC",
            "oauth_access_token": "tok",
        },
    )


def _capture_connect_listener(connector: SnowflakeConnector) -> dict:
    """Run create_engine with sqlalchemy.create_engine + the connect-event
    registration stubbed, so it doesn't need a live Snowflake (the SHOW
    DATABASES validation is fed a mock row). Returns the captured listener.
    """
    captured: dict = {}

    def fake_listens_for(target, identifier):  # noqa: ANN001, ANN202
        def deco(fn):  # noqa: ANN001, ANN202
            captured["identifier"] = identifier
            captured["fn"] = fn
            return fn

        return deco

    mock_engine = MagicMock()
    mock_engine.dialect.ischema_names = {}
    mock_engine.dialect.colspecs = {}
    # create_engine validates the database via SHOW DATABASES; db[1] is the name.
    rows = [("created_on", connector.config["database"])]
    mock_engine.connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = rows

    with (
        patch.object(connector_module.sqlalchemy, "create_engine", return_value=mock_engine),
        patch.object(connector_module.sqlalchemy.event, "listens_for", side_effect=fake_listens_for),
    ):
        engine = connector.create_engine()

    assert engine is mock_engine
    return captured


def test_connect_listener_issues_use_database():
    """Every pooled connection must run an explicit `USE DATABASE` so the
    session has a current database. Role-scoped OAuth tokens carry no default
    namespace, so without this the unqualified `CREATE TABLE` this target
    emits fails with `090105: ... does not have a current database`."""
    connector = _oauth_connector(database="CUSTOMER_DB")
    captured = _capture_connect_listener(connector)

    assert captured["identifier"] == "connect"

    cursor = MagicMock()
    dbapi_connection = MagicMock()
    dbapi_connection.cursor.return_value = cursor

    captured["fn"](dbapi_connection, MagicMock())

    cursor.execute.assert_called_once_with('USE DATABASE "CUSTOMER_DB"')
    cursor.close.assert_called_once()


def test_connect_listener_quotes_database_with_special_chars():
    """A database name containing special characters (e.g. a `$`) must be
    quoted so the USE DATABASE resolves it exactly."""
    connector = _oauth_connector(database="TEST$DB")
    captured = _capture_connect_listener(connector)

    cursor = MagicMock()
    dbapi_connection = MagicMock()
    dbapi_connection.cursor.return_value = cursor

    captured["fn"](dbapi_connection, MagicMock())

    cursor.execute.assert_called_once_with('USE DATABASE "TEST$DB"')
