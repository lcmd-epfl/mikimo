import os
import sys

def _use_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    force = os.getenv("MIKIMO_COLOR")
    if force == "1":
        return True
    if force == "0":
        return False
    term = os.getenv("TERM", "")
    return sys.stdout.isatty() and term and term.lower() != "dumb"


def _color(text: str, code: str) -> str:
    if not _use_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def configure_console():
    return None


def section(title: str) -> str:
    return _color(f"\n=== {title} ===", "1;34")


def item(label: str, value) -> str:
    label_colored = _color(str(label), "1;36")
    return f"{label_colored}: {value}"


def concentration_line(species: str, value: float) -> str:
    species_s = _color(f"[{species}]", "1;33")
    value_s = _color(f"{value:.4f}", "1;32")
    return f"{species_s} = {value_s}"
