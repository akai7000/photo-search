import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / "packaging" / "build_windows.py"
spec = importlib.util.spec_from_file_location("photo_search_build_windows", path)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class BundleValidationTests(unittest.TestCase):
    def test_build_entrypoints_reject_non_cpu_variants(self):
        root = path.parents[1]
        for script, args in (
            (root / "build_installers.py", ["gpu"]),
            (root / "setup_build_envs.py", ["gpu"]),
            (path, ["GPU", "--iscc", "unused"]),
        ):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, str(script), *args],
                                        capture_output=True, text=True, cwd=root)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid choice", result.stderr)

    def test_torchvision_binaries_are_bundled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "_C_stable.pyd").touch()
            (root / "image_stable.pyd").touch()
            (root / "jpeg8.dll").touch()
            (root / "version.py").touch()
            self.assertEqual(build.torchvision_assets(root), [
                "--add-binary", f"{root / '_C_stable.pyd'};torchvision",
                "--add-binary", f"{root / 'image_stable.pyd'};torchvision",
                "--add-binary", f"{root / 'jpeg8.dll'};torchvision",
            ])
            (root / "_C_stable.pyd").unlink()
            with self.assertRaisesRegex(RuntimeError, "native extension not found"):
                build.torchvision_assets(root)

    def test_allows_only_verified_face_models(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            models = bundle / "_internal" / "models"
            models.mkdir(parents=True)
            content = b"model"
            specs = {"one.onnx": ("one", hashlib.sha256(content).hexdigest()),
                     "two.onnx": ("two", hashlib.sha256(content).hexdigest())}
            for name in specs:
                (models / name).write_bytes(content)
            with patch.object(build, "MODEL_SPECS", specs):
                build.validate_bundle(bundle)
                unexpected = bundle / "other.onnx"
                unexpected.write_bytes(content)
                with self.assertRaisesRegex(RuntimeError, "Unexpected"):
                    build.validate_bundle(bundle)
                unexpected.unlink()
                (models / "one.onnx").write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "unexpected SHA-256"):
                    build.validate_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
