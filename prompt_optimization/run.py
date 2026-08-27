from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tiny_wordle_lab_v2.lexicon import ROOT
from tiny_wordle_lab_v2.litellm_policy import (
    DEFAULT_API_BASE,
    DEFAULT_ENV_FILE,
    read_api_key,
)

from .wordle_env import WordleAdapter


DEFAULT_CONFIG = ROOT / "prompt_optimization" / "skillopt.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    arguments, skillopt_arguments = parser.parse_known_args()

    api_key = read_api_key(arguments.env_file.expanduser())
    os.environ["OPENAI_COMPATIBLE_BASE_URL"] = arguments.api_base
    os.environ["OPENAI_COMPATIBLE_API_KEY"] = api_key

    from scripts import train as skillopt_train

    skillopt_train._ENV_REGISTRY["wordle"] = WordleAdapter
    sys.argv = [
        "skillopt-train",
        "--config",
        str(arguments.config),
        *skillopt_arguments,
    ]
    skillopt_train.main()


if __name__ == "__main__":
    main()
