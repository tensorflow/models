# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Test CMP models.
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 PYTHONPATH='.' PYOPENGL_PLATFORM=egl \
  python scripts/script_nav_agent_release.py --config_name cmp.lmap_Msc.clip5.sbpd_d_r2r+bench_test \
  --logdir output/cmp.lmap_Msc.clip5.sbpd_d_r2r

CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 PYTHONPATH='.' PYOPENGL_PLATFORM=egl \
  python scripts/script_nav_agent_release.py --config_name cmp.lmap_Msc.clip5.sbpd_rgb_r2r+bench_test \
  --logdir output/cmp.lmap_Msc.clip5.sbpd_rgb_r2r

CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 PYTHONPATH='.' PYOPENGL_PLATFORM=egl \
  python scripts/script_nav_agent_release.py --config_name cmp.lmap_Msc.clip5.sbpd_d_ST+bench_test \
  --logdir output/cmp.lmap_Msc.clip5.sbpd_d_ST

CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 PYTHONPATH='.' PYOPENGL_PLATFORM=egl \
  python scripts/script_nav_agent_release.py --config_name cmp.lmap_Msc.clip5.sbpd_rgb_ST+bench_test \
  --logdir output/cmp.lmap_Msc.clip5.sbpd_rgb_ST

CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 PYTHONPATH='.' PYOPENGL_PLATFORM=egl \
  python scripts/script_nav_agent_release.py --config_name cmp.lmap_Msc.clip5.sbpd_d_r2r_h0_64_80+bench_test \
  --logdir output/cmp.lmap_Msc.clip5.sbpd_d_r2r_h0_64_80

# Test LSTM baseline models.
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 PYTHONPATH='.' PYOPENGL_PLATFORM=egl \
  python scripts/script_nav_agent_release.py --config_name bl.v2.noclip.sbpd_d_r2r+bench_test \
  --logdir output/bl.v2.noclip.sbpd_d_r2r

CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 PYTHONPATH='.' PYOPENGL_PLATFORM=egl \
  python scripts/script_nav_agent_release.py --config_name bl.v2.noclip.sbpd_rgb_r2r+bench_test \
  --logdir output/bl.v2.noclip.sbpd_rgb_r2r

CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 PYTHONPATH='.' PYOPENGL_PLATFORM=egl \
  python scripts/script_nav_agent_release.py --config_name bl.v2.noclip.sbpd_d_ST+bench_test \
  --logdir output/bl.v2.noclip.sbpd_d_ST

CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 PYTHONPATH='.' PYOPENGL_PLATFORM=egl \
  python scripts/script_nav_agent_release.py --config_name bl.v2.noclip.sbpd_rgb_ST+bench_test \
  --logdir output/bl.v2.noclip.sbpd_rgb_ST

CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 PYTHONPATH='.' PYOPENGL_PLATFORM=egl \
  python scripts/script_nav_agent_release.py --config_name bl.v2.noclip.sbpd_d_r2r_h0_64_80+bench_test \
  --logdir output/bl.v2.noclip.sbpd_d_r2r_h0_64_80

# Visualize test trajectories in top view.
# CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 PYTHONPATH='.' PYOPENGL_PLATFORM=egl \
#   python scripts/script_plot_trajectory.py \
#     --first_person --num_steps 40 \
#     --config_name cmp.lmap_Msc.clip5.sbpd_d_r2r \
#     --imset test --alsologtostderr
