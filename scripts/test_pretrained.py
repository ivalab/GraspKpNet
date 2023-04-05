#!/usr/bin/env python
from pathlib import Path
from subprocess import run

ROOT = Path(__file__).parent.parent


def run_cornell(model_path, dataset_path, arch):
    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    cmd = " ".join(
        f"""
        python scripts/test.py dbmctdet_cornell
            --exp_id {model_path.stem}
            --arch {arch}
            --dataset cornell
            --fix_res
            --flag_test
            --load_model {model_path}
            --ae_threshold 1.0
            --ori_threshold 0.24
            --center_threshold 0.05
            --dataset_dir {dataset_path}
    """.split()
    )
    print(cmd)
    run(cmd, shell=True, cwd=ROOT)


def run_jac(model_path, dataset_path, arch):
    model_path = Path(model_path)
    dataset_path = Path(dataset_path)
    cmd = " ".join(
        f"""
        python scripts/test.py dbmctdet
            --exp_id {model_path.stem}
            --arch {arch}
            --dataset jac_coco_36
            --fix_res
            --flag_test
            --load_model {model_path}
            --ae_threshold 0.65
            --ori_threshold 0.1745
            --center_threshold 0.15
            --dataset_dir {dataset_path}
    """.split()
    )
    print(cmd)
    run(cmd, shell=True, cwd=ROOT)


if __name__ == "__main__":
    models_path = ROOT / "models"
    dataset_path = ROOT / "datasets"
    model_factory_mapping = [
        ["dla34", "dla_34"],
        ["resnet18", "res_18"],
        ["resnet50", "res_50"],
        # too large for 12gb of gpu memory
        # ["hg52", "hourglass_52"],
        # ["hg104", "hourglass_104"],
        ["vgg16", "vgg_16"],
        ["alexnet", "alex_8"],
    ]
    # check that all models are present
    for k, _ in model_factory_mapping:
        for ds in ["cornell", "ajd"]:
            path = models_path / f"model_{k}_{ds}.pth"
            assert path.exists(), f"Model {path} does not exist"

    for ds in ["cornell", "ajd"]:
        for k, v in model_factory_mapping:
            model_path = models_path / f"model_{k}_{ds}.pth"
            func = {"cornell": run_cornell, "ajd": run_jac}[ds]
            func(model_path, dataset_path, v)
