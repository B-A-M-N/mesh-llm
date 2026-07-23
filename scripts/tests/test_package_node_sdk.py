import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TARGETS = (
    "darwin-arm64",
    "darwin-x64",
    "linux-arm64",
    "linux-x64",
    "win32-x64",
)


class PackageNodeSdkTests(unittest.TestCase):
    def test_assembles_all_native_targets_and_npm_tarball(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            addons = temporary_path / "addons"
            output = temporary_path / "package"
            for target in EXPECTED_TARGETS:
                addon = addons / target / "mesh_llm_nodejs.node"
                addon.parent.mkdir(parents=True)
                addon.write_bytes(f"fixture:{target}".encode())

            subprocess.run(
                [
                    "node",
                    "scripts/package-node-sdk.mjs",
                    str(addons),
                    str(output),
                ],
                cwd=ROOT,
                check=True,
            )

            for target in EXPECTED_TARGETS:
                self.assertTrue(
                    (output / "native" / target / "mesh_llm_nodejs.node").is_file()
                )
            self.assertTrue((output / "LICENSE").is_file())

            packed = subprocess.run(
                ["npm", "pack", "--dry-run", "--json"],
                cwd=output,
                check=True,
                capture_output=True,
                text=True,
            )
            files = {entry["path"] for entry in json.loads(packed.stdout)[0]["files"]}
            for target in EXPECTED_TARGETS:
                self.assertIn(
                    f"native/{target}/mesh_llm_nodejs.node",
                    files,
                )
            self.assertIn("LICENSE", files)

    def test_rejects_a_missing_native_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            addons = temporary_path / "addons"
            output = temporary_path / "package"
            for target in EXPECTED_TARGETS[:-1]:
                addon = addons / target / "mesh_llm_nodejs.node"
                addon.parent.mkdir(parents=True)
                addon.write_bytes(b"fixture")

            result = subprocess.run(
                [
                    "node",
                    "scripts/package-node-sdk.mjs",
                    str(addons),
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("win32-x64", result.stderr)


if __name__ == "__main__":
    unittest.main()
