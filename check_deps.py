"""Simple dependency availability checker for SerpentAI-unsupervised requirements."""
import importlib
import sys
from pkgutil import find_loader

def check(pkg, alias=None):
    name = alias or pkg
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, '__version__', None)
        print(f"OK: {pkg} (module={name}) version={ver}")
    except Exception as e:
        print(f"MISSING: {pkg} -> {e}")

pkgs = [
    ('opencv-python', 'cv2'),
    ('numpy', 'numpy'),
    ('mss', 'mss'),
    ('pillow', 'PIL'),
    ('pytesseract', 'pytesseract'),
    ('pyautogui', 'pyautogui'),
    ('pyyaml', 'yaml'),
]

for pkg, mod in pkgs:
    check(pkg, mod)

print('\nPython executable:', sys.executable)
