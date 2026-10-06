"""
全局配置模块（基于 pydantic-settings）。
- 从项目根目录的 .env 读取环境变量，类型自动转换与校验（配错会在启动时报错）
- 全局超参数直接用大写变量名（如 BATCH_SIZE）
- 数据集覆盖用 "<前缀>_<大写字段名>"（如 MNIST_BATCH_SIZE），未设置则继承全局值
- 路径相对路径自动基于项目根解析为绝对路径，目录自动创建

用法：
    from config import MNIST, settings
    MNIST.data_dir                     # 数据集目录
    MNIST.hparams.batch_size           # 数据集有效超参（覆盖或全局）
    MNIST.checkpoint_dir / output_dir / log_dir / subdir
    settings.retrain                   # 强制重训开关
    settings.hparams                   # 全局超参
"""
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 超参数定义：新增超参数只需在 HParams 加一个字段
# （数据集级覆盖 MNIST_XXX 自动生效，无需其他改动）
# ============================================================
class HParams(BaseModel):
    batch_size: int = 128
    num_workers: int = 4
    latent_dim: int = 3
    learning_rate: float = 1e-3
    max_epochs: int = 20
    patience: int = 5
    save_top_k: int = 3
    seed: int = 42
    ssim_weight: float = 0.5
    cls_weight: float = 0.5


# 数据集级覆盖模型工厂：环境变量前缀为 "<前缀>_"（如 MNIST_BATCH_SIZE）
_EnvFile = PROJECT_ROOT / ".env"


def _dataset_config_cls(prefix: str, name: str) -> type[BaseSettings]:
    class _Base(BaseSettings):
        model_config = SettingsConfigDict(env_file=_EnvFile, env_prefix=prefix, extra="ignore")

    fields = {n: (f.annotation | None, None) for n, f in HParams.model_fields.items()}
    fields["subdir"] = (str | None, None)
    return create_model(name, __base__=_Base, **fields)


DatasetConfig = _dataset_config_cls("MNIST_", "MnistConfig")


# ============================================================
# 根设置：全局超参数 + 根目录 + 开关 + 各数据集覆盖
# ============================================================
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_EnvFile,
        extra="ignore",
    )

    # ---------- 根目录 ----------
    base_data_dir: Path = Path("./datasets")
    base_checkpoint_dir: Path = Path("./checkpoints")
    base_log_dir: Path = Path("./logs")
    base_output_dir: Path = Path("./outputs")

    # ---------- 其他开关 ----------
    retrain: bool = False

    # ---------- 数据集注册：新增数据集加一行 ----------
    mnist: DatasetConfig = Field(default_factory=DatasetConfig)
    # cifar10: DatasetConfig = Field(default_factory=DatasetConfig)

    # ---------- 全局超参数（拍平到根，对应 .env 中的大写变量名） ----------
    @property
    def hparams(self) -> HParams:
        return HParams(**{name: getattr(self, name) for name in HParams.model_fields})

    batch_size: int = HParams.model_fields["batch_size"].default
    num_workers: int = HParams.model_fields["num_workers"].default
    latent_dim: int = HParams.model_fields["latent_dim"].default
    learning_rate: float = HParams.model_fields["learning_rate"].default
    max_epochs: int = HParams.model_fields["max_epochs"].default
    patience: int = HParams.model_fields["patience"].default
    save_top_k: int = HParams.model_fields["save_top_k"].default
    seed: int = HParams.model_fields["seed"].default
    ssim_weight: float = HParams.model_fields["ssim_weight"].default
    cls_weight: float = HParams.model_fields["cls_weight"].default

    def dataset_hparams(self, name: str) -> HParams:
        """数据集有效超参：全局值 + 非None的数据集覆盖。"""
        overrides = getattr(self, name).model_dump(exclude={"subdir"})
        effective = {k: v for k, v in overrides.items() if v is not None}
        return HParams(**{**self.hparams.model_dump(), **effective})


settings = Settings()


# ============================================================
# 数据集视图：路径解析 + 有效超参，供业务代码直接使用
# ============================================================
def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


class Dataset:
    def __init__(self, name: str):
        self.name = name
        cfg = getattr(settings, name)
        self.subdir = cfg.subdir or name
        self.hparams: HParams = settings.dataset_hparams(name)

        self.data_dir = _resolve(settings.base_data_dir / self.subdir)
        self.checkpoint_dir = _resolve(settings.base_checkpoint_dir / self.subdir)
        self.log_dir = _resolve(settings.base_log_dir / self.subdir)
        self.output_dir = _resolve(settings.base_output_dir / self.subdir)

        for d in (self.data_dir, self.checkpoint_dir, self.log_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"Dataset({self.name!r}, subdir={self.subdir!r}, data_dir={self.data_dir})"


# 数据集单例：业务代码 `from config import MNIST`
MNIST = Dataset("mnist")


# ============================================================
# 调试入口：python config.py
# ============================================================
if __name__ == "__main__":
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"retrain = {settings.retrain}")
    print("\n[全局超参]")
    for k, v in settings.hparams.model_dump().items():
        print(f"  {k:15} = {v}")
    print(f"\n{MNIST}")
    print("  [有效超参]（括号 = 来自数据集覆盖）")
    overrides = settings.mnist.model_dump(exclude={"subdir"})
    for k, v in MNIST.hparams.model_dump().items():
        mark = " (覆盖)" if overrides.get(k) is not None else ""
        print(f"    {k:15} = {v}{mark}")

# ============================================================
# matplotlib 中文字体配置
# ============================================================
import matplotlib
import matplotlib.pyplot as plt


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
