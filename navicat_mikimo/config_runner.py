import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from navicat_mikimo.__main__ import _get_mode_handler


def _append_list_option(argv: List[str], flag: str, values):
    if values is None:
        return
    argv.append(flag)
    argv.extend([str(v) for v in values])


def build_legacy_argv_from_dict(cfg: Dict[str, Any]) -> List[str]:
    mode = cfg["mode"]
    shared = cfg.get("shared", {})
    mode_cfg = cfg.get("mode_config", {})

    argv: List[str] = [
        "-d",
        str(shared.get("dir", ".")),
        "-e",
        str(shared.get("profile_choice", 0)),
    ]
    _append_list_option(argv, "-Tf", shared.get("time"))
    _append_list_option(argv, "-t", shared.get("temp"))
    argv.extend(["-x", str(shared.get("xscale", "ls"))])
    _append_list_option(argv, "-a", shared.get("addition"))
    argv.extend(["-iq", str(shared.get("int_quality", 1))])
    argv.extend(["-ncore", str(shared.get("ncore", 1))])
    argv.extend(["-is", str(shared.get("imputer_strat", "knn"))])
    argv.extend(["-v", str(shared.get("verb", 2))])
    if shared.get("kinetic_mode"):
        argv.append("-k")

    if mode == "vp":
        argv.extend(["-xbase", str(mode_cfg.get("xbase", 20.0))])
        argv.extend(["-lm", str(mode_cfg.get("lmargin", 20))])
        argv.extend(["-rm", str(mode_cfg.get("rmargin", 20))])
        _append_list_option(argv, "-id", mode_cfg.get("idx"))
        argv.extend(["-pm", str(mode_cfg.get("plotmode", 1))])
        argv.extend(["-pq", str(mode_cfg.get("plot_quality", 1))])
        argv.extend(["-nd", str(mode_cfg.get("run_mode", 1))])
        if mode_cfg.get("percent"):
            argv.append("-p")
        if mode_cfg.get("confidence_interval"):
            argv.append("-ci")
        if mode_cfg.get("lfesr"):
            argv.append("-lfesr")
    elif mode == "cond":
        if mode_cfg.get("plot_evo"):
            argv.append("-ev")
        if mode_cfg.get("map"):
            argv.append("-tt")

    return argv


def run_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML object with keys: mode, shared, mode_config.")
    if "mode" not in cfg:
        raise ValueError("Config must include 'mode' with one of: mkm, vp, cond.")

    mode = cfg["mode"]
    handler = _get_mode_handler(mode)
    argv = build_legacy_argv_from_dict(cfg)
    return handler(argv)


def main():
    parser = argparse.ArgumentParser(
        prog="navicat_mikimo_config",
        description="Run navicat_mikimo from a YAML config file.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config containing mode/shared/mode_config.",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return run_config(str(config_path))


if __name__ == "__main__":
    main()
