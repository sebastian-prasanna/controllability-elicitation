from cycler import cycler
from matplotlib import pyplot as plt

# Categorical palette validated for CVD safety (adjacent-pair OKLab dE >= 8).
# First three slots also validate all-pairs (scatter etc.).
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]


def set_matplotlib_style():
    """Clean matplotlib style in the vein of ML / AI-safety papers.

    Sans-serif type, no top/right spines, recessive y-grid, frameless
    legends, and a colorblind-safe categorical cycle.
    """
    plt.rcParams.update({
        # figure
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        # type
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # axes & grid: recessive chrome, data forward
        "axes.facecolor": "white",
        "axes.edgecolor": "#c3c2b7",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "xtick.color": "#898781",
        "ytick.color": "#898781",
        "xtick.labelcolor": "#0b0b0b",
        "ytick.labelcolor": "#0b0b0b",
        "axes.labelcolor": "#0b0b0b",
        # marks
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "patch.linewidth": 0.5,
        "errorbar.capsize": 2,
        # legend
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "axes.prop_cycle": cycler(color=PALETTE),
    })
