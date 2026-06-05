# 可视化分析脚本

这个文件夹专门放后处理和作图脚本，和主计算 pipeline 的 `scripts/`
分开。主计算脚本负责 PuMA 求解和批量结果；这里的脚本负责把已经算好的
potential / concentration / flux field 转换成更适合论文机制分析的图和指标。

核心科学问题不是简单比较“哪里的 flux 更大”，而是比较 SE-rich 相内部的
通量是否均匀、是否被迫集中在少数瓶颈通道里。对于本项目，推荐叙事是：

WM-LPSCB 中 SE 网络分布较差，局部通量更集中，很多 SE 区域没有被充分利用，
因此归一化有效离子传输系数较低；PFDT@LPSCB 中 SE 网络更均匀，参与传输的
SE 体积分数更高，因此有效传输更好。

## SE-only 2D flux/potential 对比

```bash
python viz_scripts/01_plot_se_only_flux_potential.py --config config.json --source both
```

这个脚本读取 PuMA 保存的 field 和整数 label volume，只显示 SE-rich 相内部的
potential / concentration 与 flux magnitude。非 SE 相会被 mask 掉。WM 和 PFDT
使用相同 color scale，避免人为放大差异。

常用参数：

```bash
python viz_scripts/01_plot_se_only_flux_potential.py \
  --config config.json \
  --source whole_roi \
  --se-threshold 0.5 \
  --percentile-low 1 \
  --percentile-high 99
```

输出位置：

- `results/viz/se_only/representative/`
- `results/viz/se_only/whole_roi/`

## Whole ROI 的 SE-only 增强图

这个脚本适配当前本地下载后的目录结构：

```bash
python viz_scripts/02_plot_whole_roi_se_flux_enhanced.py
```

默认输入：

- fields: `server_results/whole_roi/bulk_fields`
- labels: `results/<sample>/label_zyx.tif`

默认输出：

- `viz/whole_roi_se_only`

输出内容包括：

- 三个中心截面上的 SE-only potential / concentration
- 三个中心截面上的 SE-only flux magnitude
- `flux / global SE median` 相对通量图
- top-flux channel overlay
- SE-only flux 分布图和 CDF
- 沿 y 方向的 plane-wise profile

2D 图默认使用论文版式：Arial，tick label 9 pt，坐标轴标题 10 pt，单 panel
轴框高度 53.97 mm，边框 0.9 pt，主刻度长度 3 pt，主刻度约 3-5 个，副刻度每个主刻度区间 1 个，
无内部网格，透明背景。

注意：potential / concentration 主要反映边界条件施加的宏观梯度，所以 WM 和
PFDT 看起来可能都接近线性变化。机制分析更应该看 flux 在 SE 内部的分布、
局域化和瓶颈，而不是只看 potential 颜色是否明显不同。

## 3D cutaway 渲染

3D 脚本使用 PyVista。该依赖是可选的，本地环境如果没有，需要先安装：

```bash
conda install -c conda-forge pyvista vtk
```

然后可以渲染类似“中间剖开一个长方体”的 SE-only 3D cutaway。默认会同时生成
whole ROI 和 30 um representative 图：

```bash
python viz_scripts/03_pyvista_whole_roi_cutaway.py
```

也可以只渲染其中一类数据或一个物理量：

```bash
python viz_scripts/03_pyvista_whole_roi_cutaway.py --dataset whole_roi --quantity flux_magnitude
python viz_scripts/03_pyvista_whole_roi_cutaway.py --dataset representative30 --quantity abs_Jy
```

`flux_magnitude` 和 `abs_Jy` 默认显示 tail map：每个样品先按自身 SE median
归一化，再用 WM+PFDT 合并后的阈值标出最低 5% 和最高 5%。中间 90% 的 SE
区域显示为灰色，低利用区域显示为蓝色，高通量集中通道显示为红色。这个默认
更适合表达 WM 中 flux 更局域化、PFDT 中极端低/高通量区域更少的机制图。
如果需要严格绝对值阈值，可以加 `--flux-tail-scale absolute`。
如果只想突出高通量 bottleneck、避免把 CAM 颗粒边界附近的低 flux 区域解释为
低利用通道，可以加：

```bash
python viz_scripts/03_pyvista_whole_roi_cutaway.py --dataset representative30 --quantity abs_Jy --flux-map-mode high_only
```

high-only 模式只保留灰色 SE 背景和红色 top tail，并输出带 `_high_only` 后缀的 PNG，
不会覆盖默认 tail map。
如果希望把 WM 的 top 5% 阈值作为共同参考，再看 PFDT 有多少 SE 区域超过这个
WM bottleneck 阈值，可以运行：

```bash
python viz_scripts/03_pyvista_whole_roi_cutaway.py --dataset representative30 --quantity abs_Jy --flux-map-mode high_only --flux-tail-reference wm
python viz_scripts/03_pyvista_whole_roi_cutaway.py --dataset representative30 --quantity flux_magnitude --flux-map-mode high_only --flux-tail-reference wm
```

这会输出 `_wm_ref_high_only.png`，适合表达“PFDT 中达到 WM 高通量拥挤程度的区域更少”。
potential 默认仍为 SE-only mask。如果需要显示 SE+AM 两相内的电势场，可以运行：

```bash
python viz_scripts/03_pyvista_whole_roi_cutaway.py --quantity potential --potential-mask se_am
```

SE+AM potential 会输出为 `_se_am_potential.png`，不覆盖原来的 `_se_potential.png`。
whole ROI 数据只保存了 `flux_magnitude`，30 um representative 数据保存了完整的
`flux` vector，因此 30 um 会额外输出沿传输方向的 `abs_Jy` 3D map。

默认输出：

- `viz/whole_roi_3d_cutaway`
- `viz/representative30_3d_cutaway`

## 几何结构 overview、contact map 和指标

几何分析统一入口：

```bash
python viz_scripts/08_analyze_geometry_3d.py
```

这个脚本同时处理 30 um representative 和 whole ROI 的整数 label volume，输出：

- 三相整体 overview 与 CAM / SE-rich / Void-carbon-rich 单相 3D cutaway
- AM-SE 与 AM-Void contact map
- phase fraction、AM-SE / AM-Void 界面面积、AM 表面接触比例、SE 连通性等几何指标

默认输出：

- `viz/geometry_3d/representative30_phase_overview_3d.png`
- `viz/geometry_3d/representative30_am_contact_adjacency_3d.png`
- `viz/geometry_3d/whole_roi_phase_overview_3d.png`
- `viz/geometry_3d/whole_roi_am_contact_adjacency_3d.png`
- `viz/geometry_metrics/geometry_metrics_summary.csv`
- `viz/geometry_metrics/geometry_contact_metrics_summary.png`

其中 AM 表面接触比例主要看：

- `am_se_area_fraction_of_am_internal_surface`
- `am_void_area_fraction_of_am_internal_surface`

这两个指标用原始 label 的 6-neighbor 面接触计数计算，不受 3D 渲染降采样影响。
默认 3D 渲染为了速度会降采样：30 um 用 4x，whole ROI 用 6x；如需更细可以调
`--representative-render-downsample` 或 `--whole-roi-render-downsample`。

旧的 AM-SE 和 AM-Void 接触邻接关系单图脚本仍可单独运行：

```bash
python viz_scripts/07_plot_contact_adjacency_3d.py
```

默认使用 30 um representative label block，输出到：

- `viz/contact_adjacency_3d`

## Whole ROI 通量不均匀性指标

运行：

```bash
python viz_scripts/04_quantify_flux_heterogeneity.py
```

默认输出：

- `viz/whole_roi_flux_heterogeneity`

这个脚本只在 SE-rich 相内部统计 flux magnitude，用来回答：

- 通量是否集中在少数 SE voxel 中？
- 是否存在大量低利用 SE 区域？
- WM 和 PFDT 哪个 SE 网络更均匀？

输出文件包括：

- `whole_roi_flux_heterogeneity_summary.csv`
- `whole_roi_flux_heterogeneity_metrics.png`
- `whole_roi_flux_lorenz_curve.png`
- `whole_roi_flux_bottleneck_y_profiles.png`
- `whole_roi_flux_low_high_map_*.png`

## 30 um representative 高分辨率指标

运行：

```bash
python viz_scripts/05_quantify_representative30_flux_heterogeneity.py
```

如果需要生成和 whole-ROI SE-only 增强图同类型的 30 um representative 图，运行：

```bash
python viz_scripts/06_plot_representative30_se_flux_enhanced.py
```

默认输入：

- `server_results/30um-in-plane/results/<sample>/representative_flux`

默认输出：

- `viz/representative30_flux_heterogeneity`

30 um representative field 保存了完整的 `flux` vector，因此这个脚本会同时分析：

- `flux_magnitude`：通量矢量模长，即 `|J|`
- `abs_Jy`：沿传输方向 y 的通量分量绝对值，即 `|J_y|`

如果要支撑“沿压片/传输方向存在瓶颈”的机制，优先看 `abs_Jy`。`flux_magnitude`
也有参考价值，但它包含横向绕流分量，不如 `J_y` 直接。

## 指标解释

以下指标都只在 SE-rich 相内部计算。这里的 flux 不是绝对离子电导率，不能写成
mS cm^-1；它用于比较相同边界条件下的归一化局部传输行为。

### `mean`

SE 内部 flux 的平均值。

它可以反映整体通量水平，但不适合单独用来解释瓶颈。因为平均值升高可能来自
网络整体更好，也可能来自少数通道局部特别高。

### `median`

SE 内部 flux 的中位数。

它比 mean 更不受少数极高 flux voxel 影响。PFDT 的 median 更高，通常说明更多
SE 区域参与了有效传输。

### `std`

SE 内部 flux 的标准差。

标准差越大，说明局部 flux 波动越强。但它受 mean 影响，所以更推荐结合 CV。

### `CV`

变异系数，计算为：

```text
CV = std / mean
```

CV 越大，说明 SE 内部 flux 相对波动越强，也就是通量分布越不均匀。

在当前结果中，WM 的 CV 高于 PFDT，说明 WM 的 SE 网络中通量更容易集中或断续。

### `Gini`

Gini 系数用于衡量 flux 分布的不均匀程度。它来自 Lorenz curve：

- `Gini = 0` 表示所有 SE voxel 承担完全相同的 flux
- `Gini` 越高，说明 flux 越集中在少数 SE voxel 中

如果 WM 的 Gini 高于 PFDT，可以解释为 WM 的 SE 网络存在更强通量局域化。

### `top_10pct_flux_share`

SE 内部 flux 最高的 10% voxel 承担的总 flux 比例。

例如 `top_10pct_flux_share = 0.24` 表示 flux 最高的 10% SE voxel 承担了约
24% 的总 SE flux。

这个指标越高，说明传输越依赖少数高通量通道。对于瓶颈机制，这是很直观的
证据：通量被挤到少数路径里，而不是均匀分布在整个 SE 网络中。

### `top_20pct_flux_share`

类似 `top_10pct_flux_share`，但统计最高 20% SE voxel 承担的总 flux 比例。

它比 top 10% 稍微稳健一些，不那么依赖极少数热点 voxel。

### `bottom_50pct_flux_share`

flux 最低的 50% SE voxel 承担的总 flux 比例。

这个指标越低，说明有大量 SE 区域虽然存在，但几乎没有参与传输，可以理解为
“低利用 SE”。如果 WM 的该指标更低，就支持 WM 中 SE 分布不理想、有效参与
传输的 SE 体积分数较少。

### `participation_ratio`

participation ratio 衡量“有效参与传输的 SE voxel 比例”。计算思想是：

```text
participation_ratio = (sum(flux)^2) / (N * sum(flux^2))
```

其中 `N` 是 SE voxel 数量。

- 越接近 1，表示 flux 更均匀地分布在 SE 网络中
- 越低，表示 flux 越集中在少数 voxel 中

因此 participation ratio 是一个“网络利用率”指标。PFDT 高于 WM，说明 PFDT
中更多 SE 区域真正参与传输。

### `localization_index_1_minus_participation`

计算为：

```text
1 - participation_ratio
```

这个指标越高，说明通量局域化越强。它和 participation ratio 表达的是同一件事，
只是方向相反：越大越差，越小越均匀。

### `p90_over_p10`

SE 内部 flux 的第 90 百分位数除以第 10 百分位数：

```text
p90_over_p10 = p90 / p10
```

这个值越大，说明高通量区域和低通量区域之间差距越大。WM 如果更高，说明 WM
同时存在更多低通量 SE 和更强的局部高通量通道，也就是更不均匀。

### `p99_over_p50`

SE 内部 flux 的第 99 百分位数除以中位数：

```text
p99_over_p50 = p99 / p50
```

这个值越大，说明极端高通量热点越强。它适合辅助识别少数非常集中的传输通道。

### `iqr_over_median`

四分位距除以中位数：

```text
iqr_over_median = (p75 - p25) / p50
```

这个指标描述中间 50% SE voxel 的相对离散程度，比 p99 等极端值更稳健。

### `fraction_below_global_median`

某个样品中，SE voxel 的 flux 低于 WM+PFDT 合并后全局中位数的比例。

如果 WM 的这个比例更高，说明 WM 中更多 SE 区域低于两组共同参考水平。

### `fraction_below_half_global_median`

某个样品中，SE voxel 的 flux 低于全局中位数一半的比例。

这个指标可以理解为“明显低利用 SE”的比例。数值越高，说明低效区域越多。

### `fraction_above_2x_global_median`

某个样品中，SE voxel 的 flux 高于全局中位数 2 倍的比例。

这个指标代表高通量热点比例。它需要和低通量比例、Gini、top-share 一起看：
如果同时存在大量低通量 SE 和少数高通量热点，通常说明通量被迫绕行或挤过瓶颈。

### `plane_mean_cv_y`

沿 y 方向逐层计算 SE 内部平均 flux，然后对这些 plane mean 求 CV。

这个指标越大，说明不同 y 截面之间的平均传输强度波动越大，可能对应沿传输方向
局部截面更容易形成瓶颈。

### `plane_total_cv_y`

沿 y 方向逐层计算 SE 内部总 flux，然后对这些 plane total 求 CV。

对于稳态传输，理想情况下总通量沿传输方向应较稳定。这个指标过大时，需要检查：

- 是否有边界层影响
- 是否 mask 或坐标方向有问题
- 是否使用的是 `flux_magnitude` 而不是传输方向分量

### `bottleneck_index_y`

whole ROI 脚本中的 y 方向瓶颈指标，计算思想是：

```text
bottleneck_index_y = 1 - min(plane_mean_flux_y) / median(plane_mean_flux_y)
```

越大表示 y 方向某些截面的 SE 平均 flux 明显低于典型水平，可能存在截面瓶颈。

### `bottleneck_index_y_interior`

30 um representative 脚本中的内部瓶颈指标。它和 `bottleneck_index_y` 类似，
但默认去掉 y 方向两端各 5% 的边界层，只看内部区域：

```text
bottleneck_index_y_interior
  = 1 - min(interior plane_mean_flux_y) / median(interior plane_mean_flux_y)
```

这个指标更适合判断材料内部网络瓶颈，避免把入口/出口边界效应误认为内部瓶颈。

### `total_bottleneck_index_y_interior`

和 `bottleneck_index_y_interior` 类似，但使用每个 y plane 的总 flux，而不是平均 flux。

在稳态场中，plane total 通常更稳定；如果这个指标很高，需要谨慎解释，优先检查
边界和 mask。机制讨论中更推荐结合 `abs_Jy` 的 `bottleneck_index_y_interior`、
Gini、top-share 和 participation ratio。

## 推荐用于论文的表述

可以写成类似：

```text
The normalized effective ionic transport coefficient is lower in WM-LPSCB.
SE-only flux analysis shows that WM-LPSCB has a higher CV, Gini coefficient,
and top-10% flux share, together with a lower participation ratio. This indicates
that ionic flux is more localized in a small fraction of the SE network, leaving
more SE-rich regions under-utilized. In contrast, PFDT@LPSCB exhibits a more
uniform flux distribution and a larger effectively participating SE network.
```

中文理解是：WM 不是所有地方都“导得低”，而是 SE 网络分布较差，导致 flux 被
集中到少数通道中，形成局部瓶颈，整体等效传输能力下降。
