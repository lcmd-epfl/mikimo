import importlib
import sys

import fire


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
    Perform microkinetic modelling (mkm) for homogeneous reactions.

    Uses energy data and coupling with volcanic for high-throughput runs.

    Run mikimo by the specified mode, which can be one of the following:
    - 'mkm': Perform a single MKM run using the top-most row in the reaction data file.
    - 'vp': Screen over all energy profiles in the reaction data file.
    - 'cond': Screen over reaction time and/or temperature.

    Usage:
    - For 'mkm' mode: navicat_mikimo mkm
    - For 'vp' mode: navicat_mikimo vp
    - For 'cond' mode: navicat_mikimo cond

    Args:
        mode (str): The mode to run [Valid choices: 'mkm', 'vp', or 'cond'].

    Returns:
        None
    """
    sys.exit(_get_mode_handler(mode)())


def main():
    fire.Fire(run)


if __name__ == "__main__":
    main()
