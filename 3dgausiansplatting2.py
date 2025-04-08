# -*- coding: utf-8 -*-
"""3DGausianSplatting2

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/bolleddu01234/3dgausiansplatting2.6467f4cf-5db4-4b74-8c67-9fcbf88dad1d.ipynb%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com/20250408/auto/storage/goog4_request%26X-Goog-Date%3D20250408T135246Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D9001489763ed3d017bb60676521314260da182e3536958ad9d898b7bdf7ddc37f08edf656bd2d065ca5e6da35c598b81ad5a1ba5bedc95d26370ca1b3fdbe6eac4d9de94f47af2e749998ed5ac8d597b6b6d0e0a08d8b4de955d2e139bfae29ebfb9c4016971e582160e6108d1ff2b4d728df196f33d81396dbe03e69c7831f570abcbe72f849921bf84be2ab951f5a043fcecb35624be01fd154ab4e803cc78ec3d544cb7abb2f1d9343a2ba419b3ac65031e65ca3ca82d33efb53fe1c8137ac595ef3665a3cc5c4dc5f72fd6bf45d47cb6b695f1238496139043358df0f94b8ed43e2a8f4fe214567241c32d79eb4940aa3c15debb563658dc19d1827efcb6
"""

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
sauravmaheshkar_nerf_dataset_path = kagglehub.dataset_download('sauravmaheshkar/nerf-dataset')
jinnywjy_tanks_and_temple_m60_colmap_preprocessed_path = kagglehub.dataset_download('jinnywjy/tanks-and-temple-m60-colmap-preprocessed')
sandeshbashyal_3d_gaussian_splatting_colmap_dataset_path = kagglehub.dataset_download('sandeshbashyal/3d-gaussian-splatting-colmap-dataset')

print('Data source import complete.')

!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install ninja imageio scipy opencv-python tqdm matplotlib
!pip install git+https://github.com/facebookresearch/pytorch3d.git
!apt-get install libgl1-mesa-glx -y

"""cài đặt"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/graphdeco-inria/gaussian-splatting.git
# %cd gaussian-splatting

# Cài đặt các module phụ thuộc
!pip install -r requirements.txt
!python setup.py install

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/dthanh02/gaussian-splatting.git
# %cd gaussian-splatting

!nvidia-smi
!apt update
!apt install cuda-toolkit-12-6

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/jonstephens85/gaussian-splatting-Windows --recursive
# %cd gaussian-splatting-Windows

!pip install torch torchvision torchaudio
!pip install imageio imageio[ffmpeg]
!pip install trimesh plyfile
!pip install git+https://github.com/NVlabs/nvdiffrast

"""cài đặt thư viện"""

!wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
!unzip tandt_db.zip -d datasets

!ls -R /kaggle/working/datasets

!apt update
!apt install -y colmap

!ls -lh /kaggle/working/gaussian-splatting/submodules/diff-gaussian-rasterization
#kiểm tra thôi

!cd /kaggle/working/gaussian-splatting && git submodule update --init --recursive

!pip install /kaggle/working/gaussian-splatting/submodules/diff-gaussian-rasterization

!pip install plyfile
#cài đặt thư viện plyfile

!pip install /kaggle/working/gaussian-splatting/submodules/simple-knn
#cài đặt simple_knn

!cp -r /kaggle/input/3d-gaussian-splatting-colmap-dataset/tandt_db/tandt /kaggle/working/dataset/
#sao chép dataset vào thư mục /kaggle/working/ trước khi train

!cp -r /kaggle/input/nerf-dataset/nerf_llff_data/nerf_llff_data /kaggle/working/dataset/
#sao chép dataset vào thư mục /kaggle/working/ trước khi train

!ls -R /kaggle/working/dataset/truck

!python /kaggle/working/gaussian-splatting/train.py \
    -s /kaggle/working/dataset/truck \
    -m /kaggle/working/output_model_1

!python /kaggle/working/gaussian-splatting/train.py \
    -s /kaggle/working/dataset/truck \
    -m /kaggle/working/output_model_1

!python /kaggle/working/gaussian-splatting/train.py \
    --source_path /kaggle/working/dataset/truck \
    --model_path /kaggle/working/output_model_2 \
    --iterations 30000 \
    --position_lr_init 0.0025 \
    --feature_lr 0.0025 \
    --opacity_lr 0.0025 \
    --scaling_lr 0.0025 \
    --rotation_lr 0.0025 \
    --percent_dense 0.01

!python /kaggle/working/gaussian-splatting/train.py \
    --source_path /kaggle/working/dataset/truck \
    --model_path /kaggle/working/output_model_3 \
    --iterations 30000 \
    --position_lr_init 0.0025 \
    --feature_lr 0.0025 \
    --opacity_lr 0.0025 \
    --scaling_lr 0.0025 \
    --rotation_lr 0.0025 \
    --percent_dense 0.012 \
    --densification_interval 150 \
    --densify_until_iter 25000

!python /kaggle/working/gaussian-splatting/train.py \
    --source_path /kaggle/working/dataset/truck \
    --model_path /kaggle/working/output_model_4 \
    --iterations 30000 \
    --position_lr_init 0.002 \
    --feature_lr 0.0025 \
    --opacity_lr 0.0025 \
    --scaling_lr 0.0025 \
    --rotation_lr 0.0025 \
    --percent_dense 0.009 \
    --densification_interval 250 \
    --densify_until_iter 28000

!python /kaggle/working/gaussian-splatting/train.py \
    --source_path /kaggle/working/dataset/truck \
    --model_path /kaggle/working/output_model_5 \
    --iterations 30000 \
    --position_lr_init 0.0015 \
    --feature_lr 0.0020 \
    --opacity_lr 0.0025 \
    --scaling_lr 0.0025 \
    --rotation_lr 0.0025 \
    --percent_dense 0.008 \
    --densification_interval 250 \
    --densify_until_iter 29000

!python train.py \
    --source_path /kaggle/working/dataset/truck \
    --model_path /kaggle/working/output_model_optimize \
    --iterations 60000 \
    --opacity_lr 0.01 \
    --densification_interval 300 \
    --percent_dense 0.015 \
    --lambda_dssim 0.05 \
    --save_iterations 7000 30000 40000 50000 60000 \
    --test_iterations 7000 30000 40000 50000 60000

!python /kaggle/working/gaussian-splatting/train.py \
    -s /kaggle/working/dataset/truck \
    -m /kaggle/working/output_model_2 \
    --iterations 60000 \
    --opacity_lr 0.01 \
    --densification_interval 300 \
    --percent_dense 0.015 \
    --lambda_dssim 0.05 \
    --save_iterations 7000 30000 40000 50000 60000 \
    --test_iterations 7000 30000 40000 50000 60000

!python /kaggle/working/gaussian-splatting/train.py \
    -s /kaggle/working/dataset/truck \
    -m /kaggle/working/output_model_1 \
    --iterations 60000 \
    --save_iterations 7000 30000 40000 50000 60000 \
    --test_iterations 7000 30000 40000 50000 60000

!python /kaggle/working/gaussian-splatting/train.py \
    --source_path /kaggle/working/dataset/truck \
    --model_path /kaggle/working/output_model_high_psnr \
    --iterations 60000 \
    --opacity_lr 0.005 \
    --densification_interval 300 \
    --percent_dense 0.02 \
    --lambda_dssim 0.05 \
    --save_iterations 7000 30000 40000 50000 60000 \
    --test_iterations 7000 30000 40000 50000 60000

!python /kaggle/working/gaussian-splatting/train.py \
    --source_path /kaggle/working/dataset/truck \
    --model_path /kaggle/working/output_model_optimized_30k \
    --iterations 30000 \
    --opacity_lr 0.005 \
    --densification_interval 200 \
    --percent_dense 0.025 \
    --lambda_dssim 0.05 \
    --save_iterations 7000 15000 30000 \
    --test_iterations 7000 15000 30000

!python /kaggle/working/gaussian-splatting/train.py \
    --source_path /kaggle/working/dataset/truck \
    --model_path /kaggle/working/output_model_optimized_30k_v2 \
    --iterations 30000 \
    --opacity_lr 0.0075 \
    --densification_interval 300 \
    --percent_dense 0.015 \
    --lambda_dssim 0.05 \
    --save_iterations 7000 15000 30000 \
    --test_iterations 7000 15000 30000

!python /kaggle/working/gaussian-splatting/train.py \
    --source_path /kaggle/working/dataset/truck \
    --model_path /kaggle/working/output_model_optimized_30k_v3 \
    --iterations 35000 \
    --opacity_lr 0.006 \
    --densification_interval 300 \
    --percent_dense 0.017 \
    --lambda_dssim 0.05 \
    --save_iterations 7000 15000 30000 35000 \
    --test_iterations 7000 15000 30000 35000

!python /kaggle/working/gaussian-splatting/train.py \
    --source_path /kaggle/working/dataset/truck \
    --model_path /kaggle/working/output_model_optimized_30k_v4 \
    --iterations 30000 \
    --opacity_lr 0.007 \
    --densification_interval 300 \
    --percent_dense 0.014 \
    --lambda_dssim 0.05 \
    --save_iterations 7000 15000 30000 \
    --test_iterations 7000 15000 30000

!python /kaggle/working/gaussian-splatting/train.py \
    --source_path /kaggle/working/dataset/truck \
    --model_path /kaggle/working/output_model_40k_full \
    --iterations 40000 \
    --opacity_lr 0.01 \
    --densification_interval 300 \
    --save_iterations 7000 30000 40000 \
    --test_iterations 7000 30000 40000

!wget -O /kaggle/working/gaussian-splatting/gaussian_renderer/__init__.py https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/gaussian_renderer/__init__.py

file_path = "/kaggle/working/gaussian-splatting/gaussian_renderer/__init__.py"

with open(file_path, "r") as f:
    content = f.readlines()

for i, line in enumerate(content):
    if "GaussianRasterizationSettings(" in line:
        print(f"{i+1}: {line.strip()}")

# !python /kaggle/working/gaussian-splatting/train.py \
#     -s /kaggle/working/datasets/db/drjohnson \
#     -m /kaggle/working/output_model_2

!zip -r /kaggle/working/output_model_1.zip /kaggle/working/output_model_1

!cp -r /kaggle/input/3d-gaussian-splatting-colmap-dataset/tandt_db/db /kaggle/working/dataset/
#sao chép dataset vào thư mục /kaggle/working/ trước khi train

!python /kaggle/working/gaussian-splatting/train.py \
    -s /kaggle/working/dataset/playroom \
    -m /kaggle/working/output_model_2

!zip -r /kaggle/working/output_model_2.zip /kaggle/working/output_model_2

!ls -R /kaggle/working/dataset/drjohnson

!python /kaggle/working/gaussian-splatting/train.py \
    -s /kaggle/working/dataset/drjohnson \
    -m /kaggle/working/output_model_3

!zip -r /kaggle/working/output_model_3.zip /kaggle/working/output_model_3

!ls -R /kaggle/working/dataset

# !git clone https://github.com/google/nerfies.git
# %cd nerfies
# !pip install -r requirements.txt
!pip install absl-py==0.13.0 flax==0.3.4 imageio==2.9.0 immutabledict==2.2.0
!pip install jax==0.2.20 numpy==1.19.5 opencv-python==4.5.3.56

pip install numpy==1.26.4

!python -c "import numpy; print(numpy.__version__)"

!pip install opencv-python==4.7.0.72

!ls /kaggle/working/dataset/nerf_llff_data/fern

import numpy as np

poses_bounds = np.load("/kaggle/working/dataset/nerf_llff_data/fern/poses_bounds.npy")
print(poses_bounds.shape)

!rm -r /kaggle/working/dataset/nerf_llff_data/fern/images_4
!rm -r /kaggle/working/dataset/nerf_llff_data/fern/images_8

!python /kaggle/working/gaussian-splatting/train.py \
    -s /kaggle/working/dataset/fern \
    -m /kaggle/working/output_model_fern