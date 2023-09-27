import sys

import fire

from . import kinetic_solver, km_volcanic, screen_reaction_condition


def run(mode):
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
