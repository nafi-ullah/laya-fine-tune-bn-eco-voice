"""Push data/hf_export/ to the Hugging Face Hub as a dataset repo (private by default, config/hf.yaml).

    .venv/bin/python scripts/10_push_hf.py

Needs HF_TOKEN (./.env) with write access to the account's repos.
"""
from __future__ import annotations

import os

from huggingface_hub import HfApi

from common import DATA, load_cfg, load_env

EXPORT = DATA / "hf_export"


def main() -> None:
    load_env()
    cfg = load_cfg("hf")
    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    repo_id = f"{user}/{cfg['repo_name']}"

    readme = EXPORT / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8").replace("{user}", user), encoding="utf-8")

    api.create_repo(repo_id, repo_type="dataset", private=cfg["private"], exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(EXPORT),
                      commit_message="Add Bangla e-commerce voice agent decision dataset")
    vis = "private" if cfg["private"] else "public"
    print(f"pushed ({vis}) → https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
