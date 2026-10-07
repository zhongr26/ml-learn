"""
配置基础设施：路径解析、公共超参、数据集配置工厂、Dataset 视图。

- 全局配置: config/global_config.py（目录、开关、公共超参默认值，读 .env）
- 项目配置: config/mnist_config.py / config/yolo_config.py（各自 HParams + env 前缀，
  读 .env.<name> 与 .env，OS 环境变量优先级最高，前缀如 MNIST_BATCH_SIZE / YOLO_IMG SZ）
"""
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from matplotlib import pyplot as plt
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EnvFile = PROJECT_ROOT / ".env"


# ============================================================
# 公共超参：各项目 HParams 的基类（字段即 .env 中的大写全局变量）
# ============================================================
class CommonHParams(BaseModel):
    batch_size: int = 64
    num_workers: int = 2
    learning_rate: float = 1e-3
    max_epochs: int = 50
    patience: int = 5
    save_top_k: int = 3
    seed: int = 42


# ============================================================
# 项目配置类工厂：为每个项目生成 "<PREFIX>_<字段>" 环境变量的覆盖配置类
# 覆盖文件 .env.<name>（可选）+ 全局 .env，OS 环境变量优先
# ============================================================
def _dataset_config_cls(name: str, prefix: str, hparams_cls: type[BaseModel]) -> type[BaseSettings]:
    from pydantic import create_model

    project_env = PROJECT_ROOT / f".env.{name}"

    class _Base(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=(project_env, _EnvFile),
            env_prefix=prefix,
            extra="ignore",
        )

    fields = {n: (Optional[f.annotation], None) for n, f in hparams_cls.model_fields.items()}
    fields["subdir"] = (Optional[str], None)
    return create_model(f"{name.capitalize()}Config", __base__=_Base, **fields)


# ============================================================
# Dataset 视图：路径解析 + 有效超参，供业务代码直接使用
# ============================================================
def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


class Dataset:
    def __init__(self, name: str, prefix: str, hparams_cls: type[BaseModel]):
        from config.global_config import settings

        self.name = name
        cfg = _dataset_config_cls(name, prefix, hparams_cls)()

        self.subdir = cfg.subdir or name
        overrides = {k: v for k, v in cfg.model_dump(exclude={"subdir"}).items() if v is not None}
        self.hparams = hparams_cls(**{**settings.common().model_dump(), **overrides})

        self.data_dir = _resolve(settings.base_data_dir / self.subdir)
        self.checkpoint_dir = _resolve(settings.base_checkpoint_dir / self.subdir)
        self.log_dir = _resolve(settings.base_log_dir / self.subdir)
        self.output_dir = _resolve(settings.base_output_dir / self.subdir)

        for d in (self.data_dir, self.checkpoint_dir, self.log_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"Dataset({self.name!r}, subdir={self.subdir!r}, data_dir={self.data_dir})"


# ============================================================
# matplotlib 中文字体配置
# ============================================================
import matplotlib


def setup_matplotlib_chinese():
    """配置 matplotlib 支持中文显示，并修复负号问题。"""
    preferred_fonts = [
        "Microsoft YaHei",  # Windows 微软雅黑
        "SimHei",  # Windows 黑体
        "PingFang SC",  # macOS 苹方
        "Heiti SC",  # macOS 黑体
        "WenQuanYi Zen Hei",  # Linux 文泉驿
        "Noto Sans CJK SC",  # Google Noto
    ]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    chosen = next((f for f in preferred_fonts if f in available), None)

    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen]
        print(f"[matplotlib] 使用中文字体: {chosen}")
    else:
        print("[matplotlib] 未找到中文字体，中文将显示为方块。")
        print("  可用字体示例:", sorted(available)[:10])

    plt.rcParams["axes.unicode_minus"] = False  # 修复负号显示
    return chosen
