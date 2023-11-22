import sys

import fire

from . import kinetic_solver, km_volcanic, screen_reaction_condition


def run(mode):
    """
    Perform microkinetic modeling (mkm) for homogeneous reaction using energy data and the coulping with volcanic to perform mkm in a high-throughput fashion.

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
    if mode == "mkm":
        sys.exit(kinetic_solver.main())
    elif mode == "vp":
        sys.exit(km_volcanic.main())
    elif mode == "cond":
        sys.exit(screen_reaction_condition.main())


def main():
    fire.Fire(run)


if __name__ == "__main__":
    main()
