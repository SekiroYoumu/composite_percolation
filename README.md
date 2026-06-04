# FIB-SEM 三相分割到 PuMA 归一化稳态传输 pipeline

这套脚本把 Dragonfly 导出的三相 binary mask 合成为整数 label volume，然后用 PuMA Python 在规则 sub-volume 上计算 normalized effective ionic transport coefficient。这里的结果不是绝对离子电导率，也不要写成 mS cm^-1；SE-rich phase 被赋予单位电导率，CAM 和 Void/carbon-rich 为近零电导率，结果用于同一边界条件下比较局部有效离子传输能力。

## 输入

默认读取 `input/`：

- WM-LPSCB: `wm-am.tiff`, `wm-se.tiff`, `wm-void.tiff`
- PFDT@LPSCB: `pfdt-am.tiff`, `pfdt-se.tiff`, `pfdt-void.tiff`

TIFF 读取后默认数组顺序为 `z_y_x`。合成的 label volume 也是整数 `z_y_x`，不会导出 RGB label：

- `1`: CAM
- `2`: SE-rich
- `3`: Void/carbon-rich
- `0`: unassigned，除非在 `config.json` 里设置 `fill_unassigned_as_void=true`

PuMA 计算前脚本会显式把 `z_y_x` 转成 `config.json` 中的 `puma_axis_order`，默认 `x_y_z`。

## 环境安装

本地或服务器：

```bash
conda env create -f environment.yml
conda activate fibsem-puma-transport
```

或：

```bash
python -m pip install -r requirements.txt
```

如果服务器上的 PuMA 需要额外编译依赖，请优先按 PuMA 官方安装方式配置好 `pumapy`，再运行本 pipeline。

注意：不要用 PyPI 的 `pip install pumapy` 安装 PuMA。PyPI 上同名包可能不是 NASA Porous Microstructure Analysis。推荐按 PuMA 官方文档使用 conda-forge：

```bash
conda create -y --name puma -c conda-forge puma
conda activate puma
```

## 配置

所有关键参数在 `config.json`：

- `voxel_size_um`: 默认 `0.07`，即 70 nm isotropic。也可改成 `[x, y, z]` 或 `{"x": ..., "y": ..., "z": ...}`。
- `through_plane_axis`: 改成实际传输方向，必须是 `x`, `y`, `z` 之一。当前默认 `y`，对应 FIB-SEM 切片平面内从图像上方到下方的方向；上一轮归档结果使用的是 `z`。
- `subvolume_um`: 默认 `[30.0, 30.0, 30.0]`，顺序是 `[x_um, y_um, z_um]`。
- `sampling_mode`: 默认 `edge_aligned`，用于 30 um 大窗口，按 `edge_aligned_counts` 在 ROI 两端/中心生成固定数量窗口。
- `edge_aligned_counts`: 默认 `x=1, y=2, z=2`，每个材料生成 4 个尽量覆盖 ROI 的 30 um 窗口。
- `stride_um`: 网格/滑窗模式使用；`edge_aligned` 模式下不用于坐标生成。
- `field_output`: 默认 `representative`，只保存 median-representative 的 3D field，以减少硬盘占用。需要所有窗口 field 时改成 `all`。
- `label_dir`: 默认 `intermediate/labels`，合并后的整数 label volume 属于中间文件，不放在 `results/`。
- `field_dtype`: 默认 `float32`，用于减小 field map 文件体积。
- `field_storage_dir`: 默认 `bulk_fields/current`，把大体积 3D field maps 放在 `results/` 外，方便快速下载轻量结果。设为 `null` 可恢复到 `results/<sample>/fields`。
- `whole_roi_field_output`: 默认 `downsampled`，整块 ROI 计算保存降采样 field；设为 `none` 可只输出 CSV。
- `whole_roi_downsample_factor`: 默认 `2`，整块 ROI field 的平均降采样倍数。
- `workers`: 默认 `18`，批量 PuMA 计算的并行进程数。64C/128G 服务器上，20 um/10 um 滑窗每组正好 18 个窗口，建议 18 个进程、每进程 1 线程。
- `smoke_test_um`: 本地 PuMA 小体积测试尺寸。
- `conductivity`: 默认 CAM=`1e-6`, SE-rich=`1.0`, Void/carbon-rich=`1e-6`。
- `side_bc`: 默认 `symmetric`，脚本会传给 PuMA 为 `s`。

## 本地验证

先合并 mask 并查看体积分数和 slice 预览：

```bash
python scripts/01_validate_and_merge_masks.py --config config.json
```

输出：

- `intermediate/labels/WM/label_zyx.tif`
- `intermediate/labels/PFDT/label_zyx.tif`
- `results/WM/label_preview_slices.png`
- `results/PFDT/label_preview_slices.png`
- `results/volume_fractions.csv`

然后跑小体积 PuMA smoke test：

```bash
python scripts/02_puma_smoke_test.py --config config.json
```

如果想强制按 voxel 数裁剪，例如 `64 x 64 x 64`：

```bash
python scripts/02_puma_smoke_test.py --config config.json --voxels 64 64 64
```

## 生成 sub-volume

```bash
python scripts/03_generate_subvolumes.py --config config.json
```

脚本使用规则网格、非重叠、只取完整 sub-volume。ROI 约 `50 x 50 x 50 um^3` 且 sub-volume 为 `25 x 25 x 25 um^3` 时，会得到 `2 x 2 x 2 = 8` 个 sub-volumes。不能整除的边缘区域暂时忽略。

如果要做滑动窗口，可以在 `config.json` 中设置 `stride_um`，或用命令行覆盖：

```bash
python scripts/03_generate_subvolumes.py --config config.json --subvolume-um 20 20 20 --stride-um 10 10 10
```

例如 `subvolume_um=20 um` 且 `stride_um=10 um` 表示 50% overlap。仍然只保留完整窗口，边缘不足一个完整窗口的区域忽略。

当前这组 ROI 在 `20 x 20 x 20 um^3` 窗口、`10 x 10 x 10 um^3` stride 下，每个材料会生成 18 个 sub-volumes，即 WM 18 个、PFDT 18 个，总计 36 个 `keff_norm` 数据点。

输出：

- `results/WM/subvolumes.csv`
- `results/PFDT/subvolumes.csv`

坐标列是 `x0,x1,y0,y1,z0,z1`，单位为 voxel index，针对 `label_zyx.tif`。

## 服务器批量运行

把整个目录上传到服务器，至少包含：

- `input/`
- `config.json`
- `scripts/`
- `requirements.txt` 或 `environment.yml`
- `run_server.sh`

建议在 tmux 中运行：

```bash
tmux new -s puma_transport
bash run_server.sh
```

`run_server.sh` 会创建 `logs/`，并按顺序运行：

1. `01_validate_and_merge_masks.py --skip-existing`
2. `02_puma_smoke_test.py`
3. `03_generate_subvolumes.py --skip-existing`
4. `04_run_puma_batch.py`
5. `05_plot_results.py`

如果 `label_zyx.tif` 或 `subvolumes.csv` 已存在，对应步骤可断点跳过。

批量计算输出：

- `results/WM/transport_results.csv`
- `results/PFDT/transport_results.csv`
- `bulk_fields/current/<sample>/fields/<subvolume_id>/transport_fields.npz`，当 `field_output=all` 时为每个选区保存
- `bulk_fields/current/<sample>/fields/<subvolume_id>/concentration_center_slice.png`
- `bulk_fields/current/<sample>/fields/<subvolume_id>/flux_magnitude_center_slice.png`
- `bulk_fields/current/<sample>/fields/<subvolume_id>/*_xy_center_z.png`
- `bulk_fields/current/<sample>/fields/<subvolume_id>/*_xz_center_y.png`
- `bulk_fields/current/<sample>/fields/<subvolume_id>/*_yz_center_x.png`
- `results/*/representative_flux/metadata.json`
- `results/*/representative_flux/median_representative_fields.npz`
- `results/*/representative_flux/flux_magnitude_center_slice.png`

median-representative sub-volume 的选择规则是 `keff_norm` 最接近该组 median，不能手动挑最好看的区域。

## 整块 ROI 计算

整块 ROI 计算使用：

```bash
python scripts/06_run_puma_whole_roi.py --config config.json --sample WM
python scripts/06_run_puma_whole_roi.py --config config.json --sample PFDT
```

服务器上可以用：

```bash
bash run_whole_roi_server.sh
```

输出：

- `results/WM/whole_roi_transport.csv`
- `results/PFDT/whole_roi_transport.csv`
- `results/whole_roi_transport.csv`
- `bulk_fields/current/<sample>/fields/whole_roi/whole_roi_downsampled_fields.npz`，当 `whole_roi_field_output=downsampled` 时保存
- `bulk_fields/current/<sample>/fields/whole_roi/*_downsampled_*center*.png`

整块 ROI field 默认只保存 `potential/concentration` 和 `flux_magnitude`，使用 `whole_roi_downsample_factor` 做平均降采样，并按 `field_dtype` 转为 `float32`。

整块 ROI downsampled field 的 WM/PFDT 对比拼图：

```bash
python scripts/08_plot_whole_roi_downsampled_comparison.py --config config.json
```

输出：

- `results/figures/whole_roi_potential_flux_comparison.png`
- `results/figures/whole_roi_potential_flux_comparison_xy_center_z.png`
- `results/figures/whole_roi_potential_flux_comparison_xz_center_y.png`
- `results/figures/whole_roi_potential_flux_comparison_yz_center_x.png`

## 作图

```bash
python scripts/05_plot_results.py --config config.json
```

如果已经保存了 median-representative field，并想补画 `xy/xz/yz` 三个正交截面：

```bash
python scripts/07_plot_representative_orthogonal_slices.py --config config.json
```

输出：

- `results/figures/keff_norm_scatter.png`
- `results/figures/phase_fraction_scatter.png`
- `results/figures/flux_magnitude_linear.png`
- `results/figures/flux_magnitude_log.png`
- `results/figures/representative_potential_orthogonal_slices.png`
- `results/figures/representative_flux_magnitude_orthogonal_slices_linear.png`
- `results/figures/representative_flux_magnitude_orthogonal_slices_log.png`
- `results/figures/representative_potential_flux_comparison.png`
- `results/figures/representative_potential_flux_comparison_xy_center_z.png`
- `results/figures/representative_potential_flux_comparison_xz_center_y.png`
- `results/figures/representative_potential_flux_comparison_yz_center_x.png`
- `results/figures/summary.csv`

散点图每个点对应一个 sub-volume，并显示 mean +/- SD。这里的 SD 描述同一个 ROI 内 sub-volume 的空间变异，不应当写成完全独立样品重复。

## PuMA API 差异

脚本优先调用：

```python
pumapy.compute_electrical_conductivity
```

如果当前 PuMA 没有 electrical conductivity 接口，会 fallback 到：

```python
pumapy.compute_thermal_conductivity
```

二者都是稳态标量传输方程；在本 pipeline 中由于 SE-rich phase 设置为单位电导率，输出解释为 normalized effective ionic transport coefficient。

如果 PuMA 版本差异导致 smoke test 报错，脚本会打印可用的 `pumapy` conductivity/workspace 相关函数。通常需要检查这些地方：

- Workspace 构造方式是否仍为 `pumapy.Workspace.from_array`
- conductivity map 类名是否为 `IsotropicConductivityMap`
- material 添加接口是否仍为 `add_material((label, label), value)`
- `compute_*_conductivity` 的参数名是否使用 `direction`, `side_bc`, `solver_type`, `tolerance`

兼容逻辑集中在 `scripts/pipeline_common.py` 的 `compute_transport()`，服务器上只需要在那里做最小修改。
