"""Verify the environment has everything the experiments need."""
import sys

MODULES = ["numpy", "pandas", "sklearn", "matplotlib", "requests"]

def main() -> int:
    print(f"python {sys.version.split()[0]}")
    missing = []
    for name in MODULES:
        try:
            mod = __import__(name)
            print(f"  ok   {name} {getattr(mod, '__version__', '?')}")
        except ImportError as exc:
            missing.append(name)
            print(f"  MISS {name}: {exc}")
    if missing:
        print(f"\ninstall with: python -m pip install {' '.join(missing)}")
        return 1
    print("\nenvironment ready")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
