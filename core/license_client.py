"""Client for validating Photo Search licenses with the licensing service."""

import json
import re
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.data_location import load_config, save_config_values

LICENSE_SERVER_URL = "https://licensing.snowfoxninja.com"
REQUEST_TIMEOUT_SECONDS = 10

_ERROR_MESSAGES = {
    "invalid_license": "This license key was not recognized.",
    "revoked": "This license has been revoked.",
    "expired": "This license has expired.",
    "limit_exceeded": "This license has no available device activations.",
    "rate_limited": "Too many license checks. Please try again in a minute.",
    "https_required": "The licensing service requires a secure connection.",
    "invalid_request": "The licensing service rejected the request.",
    "server_error": "The licensing service encountered an error.",
}


class LicenseError(Exception):
    """A license could not be activated or validated."""


class LicenseServerUnavailable(LicenseError):
    """The licensing service could not be reached to activate a license."""


@dataclass(frozen=True)
class LicenseState:
    active: bool
    message: str


def _utc_now(now: datetime | None = None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("License timestamps must include a timezone")
    return value.astimezone(timezone.utc)


def _parse_timestamp(value: object, field: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Invalid stored license {field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid stored license {field}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Invalid stored license {field}")
    return parsed.astimezone(timezone.utc)


def get_saved_license_key() -> str | None:
    key = load_config().get("license_key")
    if key is None:
        return None
    if not isinstance(key, str):
        raise ValueError("Invalid stored license key")
    key = key.strip().upper()
    return key or None


def _machine_id(config: dict) -> str:
    value = config.get("license_machine_id")
    if value is not None:
        if not isinstance(value, str):
            raise ValueError("Invalid stored license machine ID")
        try:
            return str(uuid.UUID(value))
        except ValueError as exc:
            raise ValueError("Invalid stored license machine ID") from exc
    value = str(uuid.uuid4())
    save_config_values({"license_machine_id": value})
    return value


def get_license_state() -> LicenseState:
    config = load_config()
    key = config.get("license_key")
    if key is None:
        return LicenseState(False, "Free version - photo indexing limit applies")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Invalid stored license key")

    if config.get("license_permanent") is True:
        return LicenseState(True, "License permanently activated on this device")
    last_validated = _parse_timestamp(config.get("license_last_validated"), "validation time")
    if last_validated is None:
        return LicenseState(False, "License needs to be validated online")
    return LicenseState(True, "License permanently activated on this device")


def is_licensed() -> bool:
    return get_license_state().active


def _read_json_response(response_body: bytes) -> dict:
    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LicenseError("The licensing service returned an invalid response.") from exc
    if not isinstance(payload, dict):
        raise LicenseError("The licensing service returned an invalid response.")
    return payload


def _server_error(payload: dict) -> LicenseError:
    code = payload.get("error")
    message = _ERROR_MESSAGES.get(code)
    if message is None:
        return LicenseError("The licensing service could not validate this license.")
    return LicenseError(message)


def validate_license(key: str | None = None, now: datetime | None = None) -> LicenseState:
    """Activate a key once; an already activated device never needs revalidation."""
    config = load_config()
    saved_key = get_saved_license_key()
    normalized_key = key.strip().upper() if isinstance(key, str) else ""
    if not normalized_key:
        normalized_key = saved_key or ""
    if not normalized_key:
        raise LicenseError("Enter a license key to activate Photo Search.")
    if not re.fullmatch(r"[A-Z0-9-]{1,128}", normalized_key):
        raise LicenseError("The license key format is invalid.")
    if saved_key == normalized_key and get_license_state().active:
        return get_license_state()

    machine_id = _machine_id(config)
    request = Request(
        f"{LICENSE_SERVER_URL}/api/license/validate",
        data=json.dumps({"licenseKey": normalized_key, "machineId": machine_id}).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = _read_json_response(response.read())
    except HTTPError as exc:
        payload = _read_json_response(exc.read())
        if exc.code >= 500 or exc.code == 429:
            raise LicenseServerUnavailable(
                "Could not reach the licensing service. Connect to the internet and try activation again."
            ) from exc
        raise _server_error(payload)
    except (URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise LicenseServerUnavailable(
            "Could not reach the licensing service. Connect to the internet and try activation again."
        ) from exc

    if payload.get("valid") is not True:
        if payload.get("valid") is False:
            raise _server_error(payload)
        raise LicenseError("The licensing service returned an invalid response.")

    expires_at = payload.get("expiresAt")
    parsed_expiry = _parse_timestamp(expires_at, "expiry")
    remaining_activations = payload.get("remainingActivations")
    if type(remaining_activations) is not int or remaining_activations < 0:
        raise LicenseError("The licensing service returned an invalid response.")

    checked_at = _utc_now(now)
    if parsed_expiry is not None and checked_at >= parsed_expiry:
        raise LicenseError("This license has expired.")

    save_config_values({
        "license_key": normalized_key,
        "license_machine_id": machine_id,
        "license_last_validated": checked_at.isoformat(),
        "license_expires_at": parsed_expiry.isoformat() if parsed_expiry else None,
        "license_permanent": True,
    })
    return LicenseState(True, "License permanently activated on this device")


def validate_saved_license(now: datetime | None = None) -> LicenseState:
    """Activate a saved key only if this device has not already been activated."""
    key = get_saved_license_key()
    if key is None:
        return get_license_state()
    if get_license_state().active:
        return get_license_state()
    return validate_license(key, now)
