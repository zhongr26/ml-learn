"""调试入口：python -m config"""
from config import MNIST, YOLO, PROJECT_ROOT, settings


def main():
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"retrain = {settings.retrain}")

    print("\n[全局公共超参]")
    for k, v in settings.common().model_dump().items():
        print(f"  {k:15} = {v}")

    for ds in (MNIST, YOLO):
        print(f"\n{ds}")
        print("  [有效超参]")
        for k, v in ds.hparams.model_dump().items():
            print(f"    {k:15} = {v}")


if __name__ == "__main__":
    main()
