from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class SharedCLIConfig:
    dir: str = "."
    outdir: str = "."
    quiet: bool = False
    profile_choice: int = 0
    time: Optional[List[float]] = None
    temp: Optional[List[float]] = None
    xscale: str = "ls"
    addition: Optional[List[int]] = None
    int_quality: int = 1
    ncore: int = 1
    imputer_strat: str = "knn"
    verb: int = 2
    kinetic_mode: bool = False


@dataclass
class MKMSoloConfig:
    shared: SharedCLIConfig


@dataclass
class MKMScreeningConfig:
    shared: SharedCLIConfig
    xbase: float = 20.0
    lmargin: int = 20
    rmargin: int = 20
    idx: Optional[List[int]] = None
    percent: bool = False
    plotmode: int = 1
    confidence_interval: bool = False
    lfesr: bool = False
    plot_quality: int = 1
    run_mode: int = 1


@dataclass
class MKMCondConfig:
    shared: SharedCLIConfig
    plot_evo: bool = False
    map: bool = False


CLIConfig = Union[MKMSoloConfig, MKMScreeningConfig, MKMCondConfig]


def build_config_from_namespace(args, mode: str) -> CLIConfig:
    shared = SharedCLIConfig(
        dir=args.dir,
        outdir=args.outdir,
        quiet=args.quiet,
        profile_choice=args.profile_choice,
        time=args.time,
        temp=args.temp,
        xscale=args.xscale,
        addition=args.addition,
        int_quality=args.int_quality,
        ncore=args.ncore,
        imputer_strat=args.imputer_strat,
        verb=args.verb,
        kinetic_mode=args.kinetic_mode,
    )

    if mode == "mkm_solo":
        return MKMSoloConfig(shared=shared)
    if mode == "mkm_screening":
        return MKMScreeningConfig(
            shared=shared,
            xbase=args.xbase,
            lmargin=args.lmargin,
            rmargin=args.rmargin,
            idx=args.idx,
            percent=args.percent,
            plotmode=args.plotmode,
            confidence_interval=args.confidence_interval,
            lfesr=args.lfesr,
            plot_quality=args.plot_quality,
            run_mode=args.run_mode,
        )
    if mode == "mkm_cond":
        return MKMCondConfig(shared=shared, plot_evo=args.plot_evo, map=args.map)

    raise ValueError(f"Unsupported mode for config build: {mode}")
