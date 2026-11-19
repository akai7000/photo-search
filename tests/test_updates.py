import io
import ssl
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from core import updates


BASE = "https://pub-5c62851ff87d40acaeee8eb31d082ac7.r2.dev/installers"


class UpdateTests(unittest.TestCase):
    def response(self, content=b""):
        response = io.BytesIO(content)
        response.status = 200
        return response

    def test_selects_newest_published_version_numerically(self):
        html = (
            f'<a href="{BASE}/PhotoSearch-0.9.9-CPU-Setup.exe">old</a>'
            f'<a href="{BASE}/PhotoSearch-0.10.0-CPU-Setup.exe">new</a>'
            '<a href="https://example.com/installers/PhotoSearch-9.0.0-CPU-Setup.exe">other</a>'
        ).encode()
        with patch("core.updates.urlopen", side_effect=[self.response(html), self.response()]) as request, \
             patch("core.updates.load_download_url", return_value="https://example.org/downloads"):
            result = updates.check_for_update("0.9.0")
        self.assertEqual(result, ("0.10.0", "https://example.org/downloads"))
        self.assertEqual(request.call_args_list[1].args[0].method, "HEAD")
        self.assertEqual(request.call_args_list[1].args[0].full_url,
                         f"{BASE}/PhotoSearch-0.10.0-CPU-Setup.exe")
        for call in request.call_args_list:
            self.assertEqual(call.args[0].get_header("User-agent"), "PhotoSearch/0.9.0")
            self.assertEqual(call.kwargs["context"].verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(call.kwargs["context"].check_hostname)

    def test_current_or_older_version_does_not_show_update(self):
        html = f'<a href="{BASE}/PhotoSearch-0.5.1-CPU-Setup.exe">download</a>'.encode()
        with patch("core.updates.urlopen", return_value=self.response(html)) as request:
            self.assertIsNone(updates.check_for_update("0.5.1.0"))
        self.assertEqual(request.call_count, 1)

    def test_ignores_newer_non_cpu_installer(self):
        cpu = f"{BASE}/PhotoSearch-1.0.0-CPU-Setup.exe"
        other = f"{BASE}/PhotoSearch-2.0.0-GPU-Setup.exe"
        html = f'<a href="{cpu}">cpu</a><a href="{other}">other</a>'.encode()
        with patch("core.updates.urlopen", side_effect=[self.response(html), self.response()]) as request, \
             patch("core.updates.load_download_url", return_value="https://example.org/downloads"):
            self.assertEqual(updates.check_for_update("0.5.1"), ("1.0.0", "https://example.org/downloads"))
            self.assertEqual(request.call_args_list[1].args[0].full_url, cpu)

    def test_newer_non_cpu_installer_does_not_trigger_update(self):
        cpu = f"{BASE}/PhotoSearch-1.0.0-CPU-Setup.exe"
        other = f"{BASE}/PhotoSearch-2.0.0-GPU-Setup.exe"
        html = f'<a href="{cpu}">cpu</a><a href="{other}">other</a>'.encode()
        with patch("core.updates.urlopen", return_value=self.response(html)) as request:
            self.assertIsNone(updates.check_for_update("1.0.0"))
        self.assertEqual(request.call_count, 1)

    def test_missing_newest_cpu_installer_falls_back_to_older_release(self):
        latest = f"{BASE}/PhotoSearch-2.0.0-CPU-Setup.exe"
        previous = f"{BASE}/PhotoSearch-1.0.0-CPU-Setup.exe"
        html = f'<a href="{latest}">latest</a><a href="{previous}">previous</a>'.encode()
        missing = HTTPError(latest, 404, "missing", None, None)
        with patch("core.updates.urlopen", side_effect=[self.response(html), missing, self.response()]) as request, \
             patch("core.updates.load_download_url", return_value="https://example.org/downloads"):
            self.assertEqual(updates.check_for_update("0.5.1"), ("1.0.0", "https://example.org/downloads"))
            self.assertEqual(request.call_args_list[2].args[0].full_url, previous)

    def test_missing_installers_do_not_show_update(self):
        cpu = f"{BASE}/PhotoSearch-1.0.0-CPU-Setup.exe"
        html = f'<a href="{cpu}">cpu</a>'.encode()
        missing = HTTPError(cpu, 404, "missing", None, None)
        with patch("core.updates.urlopen", side_effect=[self.response(html), missing]):
            self.assertIsNone(updates.check_for_update("0.5.1"))

    def test_local_download_url_defaults_and_rejects_unsafe_urls(self):
        with patch("core.updates.load_config", return_value={}):
            self.assertEqual(updates.load_download_url(), "https://www.snowfoxninja.com/#downloads")
        with patch("core.updates.load_config", return_value={"download_url": "https://example.org/downloads"}):
            self.assertEqual(updates.load_download_url(), "https://example.org/downloads")
        for value in ("javascript:alert(1)", "https://user:password@example.org/", None):
            with self.subTest(value=value), patch("core.updates.load_config", return_value={"download_url": value}):
                with self.assertRaises(ValueError):
                    updates.load_download_url()

    def test_network_failure_is_reported_to_caller(self):
        with patch("core.updates.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(URLError):
                updates.check_for_update("0.5.1")


if __name__ == "__main__":
    unittest.main()
