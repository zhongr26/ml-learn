"""
全局配置：根目录、开关、公共超参默认值。只读 .env（无前缀大写变量）。
"""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.base import CommonHParams, _EnvFile


class GlobalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_EnvFile, extra="ignore")

    # ---------- 根目录 ----------
    base_data_dir: Path = Path("./datasets")
    base_checkpoint_dir: Path = Path("./checkpoints")
    base_log_dir: Path = Path("./logs")
    base_output_dir: Path = Path("./outputs")

    # ---------- 开关 ----------
    retrain: bool = False

    # ---------- 公共超参默认值（对应 .env 大写变量名） ----------
    batch_size: int = CommonHParams.model_fields["batch_size"].default
    num_workers: int = CommonHParams.model_fields["num_workers"].default
    learning_rate: float = CommonHParams.model_fields["learning_rate"].default
    max_epochs: int = CommonHParams.model_fields["max_epochs"].default
    patience: int = CommonHParams.model_fields["patience"].default
    save_top_k: int = CommonHParams.model_fields["save_top_k"].default
    seed: int = CommonHParams.model_fields["seed"].default

    def common(self) -> CommonHParams:
        return CommonHParams(**{n: getattr(self, n) for n in CommonHParams.model_fields})


settings = GlobalSettings()
