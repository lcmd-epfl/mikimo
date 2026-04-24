import importlib
import sys

import fire

BANNER = r"""
███    ███ ██ ██   ██ ██ ███    ███  ██████
████  ████ ██ ██  ██  ██ ████  ████ ██    ██
██ ████ ██ ██ █████   ██ ██ ████ ██ ██    ██
██  ██  ██ ██ ██  ██  ██ ██  ██  ██ ██    ██
██      ██ ██ ██   ██ ██ ██      ██  ██████
"""


def _get_mode_handler(mode):
    modules = {
        "mkm": "navicat_mikimo.kinetic_solver_runner",
        "vp": "navicat_mikimo.km_volcanic",
        "cond": "navicat_mikimo.screen_reaction_condition",
    }
    if mode not in modules:
        raise ValueError(
            f"Unknown mode '{mode}'. Valid modes are: mkm, vp, cond."
        )
    module = importlib.import_module(modules[mode])
    return module.main


def run(mode):
    """
    MIKIMO: microkinetic modeling toolkit

    Perform microkinetic modeling for homogeneous catalytic reactions.

    Modes:
    - mkm  : single profile simulation and concentration evolution
    - vp   : profile screening, volcano plots, activity/selectivity maps
    - cond : reaction-condition screening over time and/or temperature

    Quick start:
    - navicat_mikimo mkm  -d test_cases/test_1 -t 298.15 -Tf 60
    - navicat_mikimo vp   -d test_cases/pincer_CO2 -t 323.15 -Tf 7200 -nd 0
    - navicat_mikimo cond -d test_cases/pincer_CO2_jacs -Tf 7200 7300 -t 298.15

    Args:
        mode (str): One of: mkm, vp, cond.

    Returns:
        None
    """
    sys.exit(_get_mode_handler(mode)())


def main():
    print(BANNER)
    fire.Fire(run)


if __name__ == "__main__":
    main()
