"""
全局配置模块。
- 从项目根目录的 .env 读取环境变量
- 基于"根目录 + 数据集子目录"推导各数据集路径
- 自动创建所有必要的目录
- 支持数据集级别的超参数覆盖（如 MNIST_BATCH_SIZE 覆盖 BATCH_SIZE）
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ============================================================
# 项目根目录 = 本文件所在目录
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

# 显式指定 .env 路径，避免受当前工作目录影响
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# 类型转换与路径解析辅助函数
# ============================================================
def _resolve_path(raw: str) -> Path:
    """相对路径基于项目根解析为绝对路径，绝对路径原样返回。"""
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _env_path(key: str, default: str) -> Path:
    return _resolve_path(os.getenv(key, default))


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


# ============================================================
# 根目录
# ============================================================
BASE_DATA_DIR = _env_path("BASE_DATA_DIR", "./datasets")
BASE_CHECKPOINT_DIR = _env_path("BASE_CHECKPOINT_DIR", "./checkpoints")
BASE_LOG_DIR = _env_path("BASE_LOG_DIR", "./logs")
BASE_OUTPUT_DIR = _env_path("BASE_OUTPUT_DIR", "./outputs")


# ============================================================
# 数据集路径工厂
# ============================================================
def dataset_dirs(subdir: str) -> dict[str, Path]:
    """给定数据集子目录，返回该数据集的四个标准目录。"""
    return {
        "data": BASE_DATA_DIR / subdir,
        "checkpoint": BASE_CHECKPOINT_DIR / subdir,
        "log": BASE_LOG_DIR / subdir,
        "output": BASE_OUTPUT_DIR / subdir,
    }


# ---------- MNIST ----------
MNIST_SUBDIR = os.getenv("MNIST_SUBDIR", "mnist")
_MNIST = dataset_dirs(MNIST_SUBDIR)
MNIST_DATA_DIR = _MNIST["data"]
MNIST_CHECKPOINT_DIR = _MNIST["checkpoint"]
MNIST_LOG_DIR = _MNIST["log"]
MNIST_OUTPUT_DIR = _MNIST["output"]

# ---------- 新增数据集时照抄上面 5 行即可 ----------
# CIFAR10_SUBDIR       = os.getenv("CIFAR10_SUBDIR", "cifar10")
# _CIFAR10             = dataset_dirs(CIFAR10_SUBDIR)
# CIFAR10_DATA_DIR       = _CIFAR10["data"]
# CIFAR10_CHECKPOINT_DIR = _CIFAR10["checkpoint"]
# CIFAR10_LOG_DIR        = _CIFAR10["log"]
# CIFAR10_OUTPUT_DIR     = _CIFAR10["output"]


# ============================================================
# 全局训练超参数
# ============================================================
BATCH_SIZE = _env_int("BATCH_SIZE", 128)
NUM_WORKERS = _env_int("NUM_WORKERS", 4)
LATENT_DIM = _env_int("LATENT_DIM", 3)
LEARNING_RATE = _env_float("LEARNING_RATE", 1e-3)
MAX_EPOCHS = _env_int("MAX_EPOCHS", 20)
PATIENCE = _env_int("PATIENCE", 5)
SAVE_TOP_K = _env_int("SAVE_TOP_K", 3)
SEED = _env_int("SEED", 42)


# ============================================================
# 数据集级别的超参数（未设置则回退到全局值）
# ============================================================
def dataset_hparams(prefix: str) -> dict:
    """
    返回以 prefix 开头的超参数，未设置时回退到全局值。
    例: dataset_hparams("MNIST") 会尝试读取 MNIST_BATCH_SIZE 等。
    """
    return {
        "batch_size": _env_int(f"{prefix}_BATCH_SIZE", BATCH_SIZE),
        "num_workers": _env_int(f"{prefix}_NUM_WORKERS", NUM_WORKERS),
        "latent_dim": _env_int(f"{prefix}_LATENT_DIM", LATENT_DIM),
        "learning_rate": _env_float(f"{prefix}_LEARNING_RATE", LEARNING_RATE),
        "max_epochs": _env_int(f"{prefix}_MAX_EPOCHS", MAX_EPOCHS),
        "patience": _env_int(f"{prefix}_PATIENCE", PATIENCE),
        "save_top_k": _env_int(f"{prefix}_SAVE_TOP_K", SAVE_TOP_K),
        "seed": _env_int(f"{prefix}_SEED", SEED),
    }


MNIST_HPARAMS = dataset_hparams("MNIST")

# ============================================================
# 自动创建目录
# ============================================================
_ALL_DIRS = (
    BASE_DATA_DIR, BASE_CHECKPOINT_DIR, BASE_LOG_DIR, BASE_OUTPUT_DIR,
    MNIST_DATA_DIR, MNIST_CHECKPOINT_DIR, MNIST_LOG_DIR, MNIST_OUTPUT_DIR,
)
for _d in _ALL_DIRS:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 调试入口：python config.py
# ============================================================
if __name__ == "__main__":
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print("\n[根目录]")
    print(f"  BASE_DATA_DIR       = {BASE_DATA_DIR}")
    print(f"  BASE_CHECKPOINT_DIR = {BASE_CHECKPOINT_DIR}")
    print(f"  BASE_LOG_DIR        = {BASE_LOG_DIR}")
    print(f"  BASE_OUTPUT_DIR     = {BASE_OUTPUT_DIR}")
    print("\n[MNIST]")
    print(f"  DATA_DIR       = {MNIST_DATA_DIR}")
    print(f"  CHECKPOINT_DIR = {MNIST_CHECKPOINT_DIR}")
    print(f"  LOG_DIR        = {MNIST_LOG_DIR}")
    print(f"  OUTPUT_DIR     = {MNIST_OUTPUT_DIR}")
    print("\n[全局超参]")
    print(f"  BATCH_SIZE={BATCH_SIZE}  MAX_EPOCHS={MAX_EPOCHS}  SEED={SEED}")
    print("\n[MNIST 有效超参]")
    for k, v in MNIST_HPARAMS.items():
        print(f"  {k:15s} = {v}")

# ============================================================
# matplotlib 中文字体配置
# ============================================================
import matplotlib
import matplotlib.pyplot as plt


def setup_matplotlib_chinese():
    """配置 matplotlib 支持中文显示，并修复负号问题。"""
    # 按优先级尝试可用字体
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


# ============================================================
# 对外暴露
# ============================================================
__all__ = [
    "PROJECT_ROOT",
    # 根目录
    "BASE_DATA_DIR", "BASE_CHECKPOINT_DIR", "BASE_LOG_DIR", "BASE_OUTPUT_DIR",
    # MNIST
    "MNIST_DATA_DIR", "MNIST_CHECKPOINT_DIR", "MNIST_LOG_DIR", "MNIST_OUTPUT_DIR",
    "MNIST_HPARAMS",
    # 全局超参
    "BATCH_SIZE", "NUM_WORKERS", "LATENT_DIM", "LEARNING_RATE",
    "MAX_EPOCHS", "PATIENCE", "SAVE_TOP_K", "SEED",
    # 工具
    "dataset_dirs", "dataset_hparams",
    "setup_matplotlib_chinese"
]
