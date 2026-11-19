import io
import json
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from core import data_location, license_client


class JsonResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.payload


class LicenseClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patcher = patch.object(data_location, "CONFIG_PATH", Path(self.temp.name) / "config.json")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_validation_posts_key_and_persistent_machine_id_to_configured_service(self):
        now = datetime(2026, 11, 18, 22, 0, tzinfo=timezone.utc)
        response = JsonResponse({
            "valid": True,
            "expiresAt": "2027-01-01T00:00:00Z",
            "remainingActivations": 2,
        })
        with patch("core.license_client.urlopen", return_value=response) as urlopen:
            state = license_client.validate_license(" aaaa-bbbb-cccc-dddd ", now)
            saved_machine_id = license_client.load_config()["license_machine_id"]
            next_year = license_client.validate_saved_license(now + timedelta(days=400))

        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://licensing.snowfoxninja.com/api/license/validate",
        )
        self.assertEqual(json.loads(request.data), {
            "licenseKey": "AAAA-BBBB-CCCC-DDDD",
            "machineId": saved_machine_id,
        })
        self.assertEqual(str(uuid.UUID(saved_machine_id)), saved_machine_id)
        urlopen.assert_called_once()
        config = data_location.load_config()
        self.assertEqual(config["license_key"], "AAAA-BBBB-CCCC-DDDD")
        self.assertEqual(config["license_last_validated"], now.isoformat())
        self.assertEqual(config["license_expires_at"], "2027-01-01T00:00:00+00:00")
        self.assertTrue(config["license_permanent"])
        self.assertTrue(state.active)
        self.assertTrue(next_year.active)

    def test_permanent_activation_stays_valid_offline_indefinitely(self):
        now = datetime(2026, 11, 18, 22, 0, tzinfo=timezone.utc)
        data_location.save_config_values({
            "license_key": "AAAA-BBBB-CCCC-DDDD",
            "license_machine_id": "d4c393eb-67cd-426f-97fb-4c2641052be8",
            "license_last_validated": (now - timedelta(days=3650)).isoformat(),
            "license_expires_at": (now - timedelta(days=3649)).isoformat(),
            "license_permanent": True,
        })
        with patch("core.license_client.urlopen", side_effect=AssertionError("must not revalidate")) as urlopen:
            state = license_client.validate_saved_license(now)
        self.assertTrue(state.active)
        self.assertIn("permanently", state.message)
        urlopen.assert_not_called()

    def test_legacy_successful_validation_migrates_to_permanent_activation(self):
        now = datetime(2026, 11, 18, 22, 0, tzinfo=timezone.utc)
        data_location.save_config_values({
            "license_key": "AAAA-BBBB-CCCC-DDDD",
            "license_machine_id": "d4c393eb-67cd-426f-97fb-4c2641052be8",
            "license_last_validated": (now - timedelta(days=365)).isoformat(),
            "license_expires_at": (now - timedelta(days=1)).isoformat(),
        })
        self.assertTrue(license_client.is_licensed())

    def test_existing_activation_ignores_later_revocation_and_invalid_other_keys(self):
        now = datetime(2026, 11, 18, 22, 0, tzinfo=timezone.utc)
        data_location.save_config_values({
            "license_key": "AAAA-BBBB-CCCC-DDDD",
            "license_machine_id": "d4c393eb-67cd-426f-97fb-4c2641052be8",
            "license_last_validated": now.isoformat(),
            "license_expires_at": None,
            "license_permanent": True,
        })
        state = license_client.validate_license("AAAA-BBBB-CCCC-DDDD", now)
        self.assertTrue(state.active)

        invalid_new_key = HTTPError(
            license_client.LICENSE_SERVER_URL,
            403,
            "Forbidden",
            {},
            io.BytesIO(b'{"valid":false,"error":"invalid_license"}'),
        )
        with patch("core.license_client.urlopen", side_effect=invalid_new_key):
            with self.assertRaisesRegex(license_client.LicenseError, "not recognized"):
                license_client.validate_license("EEEE-FFFF-GGGG-HHHH", now)
        self.assertTrue(license_client.is_licensed())
        self.assertEqual(license_client.get_saved_license_key(), "AAAA-BBBB-CCCC-DDDD")

    def test_offline_first_activation_does_not_unlock(self):
        with patch("core.license_client.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(license_client.LicenseServerUnavailable):
                license_client.validate_license("AAAA-BBBB-CCCC-DDDD")
        self.assertFalse(license_client.is_licensed())


if __name__ == "__main__":
    unittest.main()
