#!/usr/bin/env python3
"""Check project readiness for Stable Diffusion Web UI."""

import importlib.util
import sys
from pathlib import Path

REQUIRED_PYTHON_VERSIONS = [(3, 10), (3, 11)]
REQUIRED_PACKAGES = ["gradio", "fastapi", "numpy", "torch"]
MODEL_DIR = Path("models") / "Stable-diffusion"
EXTENSION_DIR = Path("extensions") / "pod_design_tool"


def check_python_version():
    version = sys.version_info
    supported = any(version.major == major and version.minor == minor for major, minor in REQUIRED_PYTHON_VERSIONS)
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    if supported:
        print("  - Python version is supported.")
        return True

    supported_str = ", ".join(f"{major}.{minor}" for major, minor in REQUIRED_PYTHON_VERSIONS)
    print(f"  - Unsupported Python version. Recommended: {supported_str}.")
    return False


def check_package(name):
    spec = importlib.util.find_spec(name)
    if spec is not None:
        print(f"{name}: installed")
        return True

    print(f"{name}: missing")
    return False


def check_model_weights():
    if not MODEL_DIR.exists() or not MODEL_DIR.is_dir():
        print(f"Model directory missing: {MODEL_DIR}")
        return False

    candidates = list(MODEL_DIR.glob("*.ckpt")) + list(MODEL_DIR.glob("*.safetensors"))
    if candidates:
        print(f"Model checkpoint found: {candidates[0].name}")
        return True

    print(f"No .ckpt or .safetensors files found in {MODEL_DIR}")
    return False


def check_extension():
    if not EXTENSION_DIR.exists() or not EXTENSION_DIR.is_dir():
        print(f"Extension directory missing: {EXTENSION_DIR}")
        return False

    metadata = EXTENSION_DIR / "metadata.ini"
    script = EXTENSION_DIR / "scripts" / "pod_design_tool.py"
    ok = True

    if not metadata.exists():
        print(f"Missing extension metadata file: {metadata}")
        ok = False
    else:
        print(f"Extension metadata found: {metadata.name}")

    if not script.exists():
        print(f"Missing extension script file: {script}")
        ok = False
    else:
        print(f"Extension script found: {script.name}")

    return ok


def run_checks():
    print("Project readiness check for stable-diffusion-webui")
    print("=" * 50)

    python_ok = check_python_version()
    print()

    print("Dependency checks:")
    packages_ok = True
    for package in REQUIRED_PACKAGES:
        packages_ok = check_package(package) and packages_ok
    print()

    weights_ok = check_model_weights()
    print()

    extension_ok = check_extension()
    print()

    print("Summary")
    print("- Python version supported: ", "OK" if python_ok else "FAILED")
    print("- Required packages installed: ", "OK" if packages_ok else "FAILED")
    print("- Model weights available: ", "OK" if weights_ok else "FAILED")
    print("- POD Design Tool extension files: ", "OK" if extension_ok else "FAILED")
    print("=" * 50)

    if python_ok and packages_ok and weights_ok and extension_ok:
        print("All checks passed. The project looks ready to run.")
        return 0

    print("One or more checks failed. Fix the issues above before starting the UI.")
    print("Recommended fixes:")
    if not python_ok:
        print("  - install Python 3.10 or 3.11 and re-run the check")
    if not packages_ok:
        print("  - install missing dependencies with: python -m pip install -r requirements.txt")
    if not weights_ok:
        print("  - place a Stable Diffusion model checkpoint in models/Stable-diffusion/")
    if not extension_ok:
        print("  - verify the extensions/pod_design_tool folder structure and metadata files")
    return 1


if __name__ == "__main__":
    raise SystemExit(run_checks())
