import os
from neuralhydrology.utils.config import Config
from pathlib import Path


def _main():
    directory = "configs/transformer_gridsearch"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
        config = Config(Path(directory,filename))
        config._cfg["train_basin_file"] = "basin_lists/100_basins.txt"
        config.dump_config(folder=Path("configs/new_Transformer_gridsearch"),
                           filename=filename)

if __name__ == "__main__":
    _main()
