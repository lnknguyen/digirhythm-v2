from config import PATHS

from .screen import ScreenProcessor
from .sleep import SleepProcessor
import hydra
from omegaconf import DictConfig, OmegaConf

DATA_PATH = "data/interim/dtu/"


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg.processor))

    sensor = cfg.processor.sensor
    frequency = cfg.processor.frequency

    dfs = []

    if sensor == "screen":
        processor = ScreenProcessor(
            sensor_name="screen",
            path=PATHS["dtu"]["dtuawarescreen"],
            frequency=frequency,
        )
    elif sensor == "sleep":
        processor = SleepProcessor(
            sensor_name="sleep",
            path=PATHS["dtu"]["dtuawarescreen"],
            frequency=frequency,
        )
    else:
        raise ValueError(
            "Invalid processor type. Please choose for this list: [acti, screen, sms,or call, battery, accelerometer]"
        )

    res = processor.extract_features().reset_index()

    # Keep only the unique combinations of 'id', 'question', 'answer', and 'choice_text'
    res.to_csv(f"{DATA_PATH}/{sensor}_{frequency}.csv", index=False)


if __name__ == "__main__":
    main()
