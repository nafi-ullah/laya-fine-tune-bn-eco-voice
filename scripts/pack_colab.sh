#!/usr/bin/env bash
# Bundle what Colab needs (tokenized items + training code) into dist/laya-colab.zip.
# Upload the zip to Google Drive → MyDrive/laya-fine-tune/, then run notebooks/colab_train.ipynb.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist/laya-colab && mkdir -p dist/laya-colab/scripts dist/laya-colab/data/build
cp scripts/common.py scripts/06_preprocess.py scripts/train.py dist/laya-colab/scripts/
mkdir -p dist/laya-colab/config && cp config/*.yaml dist/laya-colab/config/
cp requirements-train.txt dist/laya-colab/
cp data/build/train_items*.pt data/build/calib_items*.pt dist/laya-colab/data/build/
(cd dist && rm -f laya-colab.zip && zip -qr laya-colab.zip laya-colab)
du -sh dist/laya-colab.zip
