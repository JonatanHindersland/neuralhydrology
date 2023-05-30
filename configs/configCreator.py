import argparse
from pathlib import Path

from neuralhydrology.utils.config import Config

def _get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--variable', type=str)
    parser.add_argument('--start', type=int)
    parser.add_argument('--stop', type=int)
    parser.add_argument('--interval', type=int)
    parser.add_argument('--multiplier', type=int)
    args = vars(parser.parse_args())

    return args

def _main():
    args = _get_args()
    config = Config(Path(args["config_file"]))
    x = args["start"]
    while(x<=args["stop"]):
        config._cfg[args["variable"]] = x
        config._cfg["experiment_name"] = f"FullTransformer{args['variable']}{x}"
        config.dump_config(folder=Path("configs/Transformer_gridsearch"),
                           filename=f"FullTransformer{args['variable']}{x}.yml")
        if args["multiplier"] != 0:
            x = x*args["multiplier"]
        else:
            x = x+args["interval"]

if __name__ == "__main__":
    _main()
