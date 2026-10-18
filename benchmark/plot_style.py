"""Shared publication-quality plotting styles for benchmark figures."""

import string
from pathlib import Path

import matplotlib as mpl

# =========================================================
# 一、统一配色预设
# =========================================================
# 说明：
# 这里保留三套常用的离散配色预设：
#
# 1. default
#    作为“通用默认配色”，区分度最好，适合大多数论文图，尤其是多组曲线/多类别对比图。
#
# 2. science_like
#    更克制一些，适合作为类似 Science 风格的备用配色。
#
# 3. nature_like
#    颜色更柔和，适合作为类似 Nature/综合期刊正文主图的备用配色。
#
# 每套配色除了 colors（主色列表）以外，还额外提供：
# - accent: 重点强调色，常用于中位线、文字或辅助元素
# - gray: 中性灰，常用于参考线、辅助轴元素
# - light_gray: 更浅的灰色，常用于次要元素或背景元素
PALETTES = {
    "default": {
        "colors": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"],
        "accent": "#000000",
        "gray": "#666666",
        "light_gray": "#BDBDBD",
    },
    "science_like": {
        "colors": ["#2F5D8A", "#D98C2B", "#3C8D7D", "#8C6BB1", "#C85A5A"],
        "accent": "#222222",
        "gray": "#666666",
        "light_gray": "#BDBDBD",
    },
    "nature_like": {
        "colors": ["#2F5D8A", "#D9A441", "#3C8D7D", "#8C6BB1", "#C96A5A"],
        "accent": "#222222",
        "gray": "#666666",
        "light_gray": "#BDBDBD",
    },
}

DEFAULT_PALETTE_NAME = "default"

# =========================================================
# 二、统一线型与 marker 顺序
# =========================================================
# 说明：
# 当图中需要同时出现多条曲线时，仅靠颜色有时不够稳妥。
# 因此建议固定一套“颜色 + 线型 + marker”的共同编码方式。
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
MARKERS = ["o", "s", "^", "D", "v"]

# =========================================================
# 三、统一尺寸常量
# =========================================================
# 说明：
# 为了更符合大家的直觉，并方便与 Adobe Illustrator 等软件中的尺寸设置对应，
# 这里优先使用“厘米（cm）”作为尺寸单位，再自动换算成英寸（inch）。
#
# 换算关系：
# 1 inch = 2.54 cm
CM_TO_INCH = 1.0 / 2.54


def cm_to_inch(width_cm, height_cm):
    """
    将厘米尺寸换算为英寸尺寸。

    参数
    ----
    width_cm, height_cm : float
        宽度和高度，单位为厘米。

    返回
    ----
    tuple[float, float]
        matplotlib 可直接使用的 (宽, 高) 英寸尺寸。
    """
    return (width_cm * CM_TO_INCH, height_cm * CM_TO_INCH)


# 常用尺寸（先给出厘米值，再自动换算为英寸）
# 这些厘米值更方便与排版和 Illustrator 中的尺寸对应。
SINGLE_COLUMN_CM = (8.5, 5.8)
DOUBLE_COLUMN_CM = (17.4, 10.7)
TRIPLE_ROW_CM = (17.4, 5.6)
TRIPTYCH_CM = (26.2, 7.5)

# matplotlib 实际使用的英寸尺寸
SINGLE_COLUMN = cm_to_inch(*SINGLE_COLUMN_CM)
DOUBLE_COLUMN = cm_to_inch(*DOUBLE_COLUMN_CM)
TRIPLE_ROW = cm_to_inch(*TRIPLE_ROW_CM)
TRIPTYCH = cm_to_inch(*TRIPTYCH_CM)

# =========================================================
# 四、获取配色的辅助函数
# =========================================================
def get_palette(name=DEFAULT_PALETTE_NAME):
    """
    根据名称返回整套配色字典。
    """
    if name not in PALETTES:
        raise ValueError(f"Unknown palette: {name}. Available: {list(PALETTES)}")
    return PALETTES[name]


def get_palette_colors(name=DEFAULT_PALETTE_NAME):
    """
    只返回某套配色中的主色列表。
    """
    return get_palette(name)["colors"]


# =========================================================
# 五、统一全局作图风格
# =========================================================
def set_paper_style(base_fontsize=7):
    """
    设置统一的论文作图风格（基于 matplotlib.rcParams）。
    """
    mpl.rcParams.update(
        {
            # 字体相关设置
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": base_fontsize,
            "axes.labelsize": base_fontsize,
            "axes.titlesize": base_fontsize,
            "xtick.labelsize": base_fontsize - 0.5,
            "ytick.labelsize": base_fontsize - 0.5,
            "legend.fontsize": base_fontsize - 0.8,
            "legend.title_fontsize": base_fontsize - 0.5,
            "axes.unicode_minus": True,
            "text.usetex": False,
            "mathtext.fontset": "dejavusans",
            # 线条 / marker / 误差棒
            "lines.linewidth": 1.15,
            "lines.markersize": 3.4,
            "errorbar.capsize": 2.0,
            # 坐标轴外观
            "axes.linewidth": 0.6,
            "axes.labelpad": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # 刻度样式
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "xtick.major.width": 0.6,
            "ytick.major.size": 3.5,
            "ytick.major.width": 0.6,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            # 图例样式
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.45,
            "legend.borderaxespad": 0.25,
            "legend.columnspacing": 0.75,
            # 图像与导出相关
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# =========================================================
# 六、子图标签相关函数
# =========================================================
def format_panel_label(index, style="upper"):
    """
    根据索引生成子图标签文本。
    style 可选：
    - upper       -> A, B, C, ...
    - lower       -> a, b, c, ...
    - paren       -> (a), (b), (c), ...
    - paren_upper -> (A), (B), (C), ...
    """
    if index < 0:
        raise ValueError("index must be non-negative")
    if index >= 26:
        raise ValueError("This formatter supports indices 0-25 only.")

    if style == "upper":
        return string.ascii_uppercase[index]
    elif style == "lower":
        return string.ascii_lowercase[index]
    elif style == "paren":
        return f"({string.ascii_lowercase[index]})"
    elif style == "paren_upper":
        return f"({string.ascii_uppercase[index]})"

    raise ValueError(f"Unknown style: {style}")


def add_panel_label(
    ax,
    label=None,
    index=None,
    style="upper",
    x=-0.12,
    y=1.02,
    fontsize=9,
    fontweight="normal",
):
    """
    在指定坐标轴 ax 上添加子图标签。
    默认不加粗。
    """
    if label is None:
        if index is None:
            raise ValueError("Provide either label or index.")
        label = format_panel_label(index, style=style)

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        va="bottom",
        ha="left",
    )


# =========================================================
# 七、统一导出函数
# =========================================================
def export_figure(fig, filename, outdir="figures", formats=("pdf", "png"), dpi=600):
    """
    统一导出图形文件。
    默认同时导出 PDF 和 PNG。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    saved = []
    for fmt in formats:
        path = outdir / f"{filename}.{fmt}"

        if fmt.lower() in ("png", "jpg", "jpeg", "tif", "tiff"):
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
        else:
            fig.savefig(path, bbox_inches="tight")

        saved.append(path)

    return saved
