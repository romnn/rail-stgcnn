import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

colors = dict(
    green="#a8ff80",
    orange="#ff7b00",
    yellow="#ebe700",
    pink="#e32d73",
    blue="#89fffd",
    pink2="#ef32d9",
    red="#f54242",
)


def delay_heat_cmap(vmin=-1, vmax=1, center=None):
    assert vmin <= 0 <= vmax
    if center is None:
        center = abs(vmin) / (abs(vmin) + vmax)
    else:
        center = (abs(vmin) + center) / (abs(vmin) + vmax)
    # This is broken
    palette = [
        (0, mcolors.colorConverter.to_rgba(colors["green"], alpha=1)),
        (center, mcolors.colorConverter.to_rgba("black", alpha=1)),
        (1, mcolors.colorConverter.to_rgba(colors["red"], alpha=1)),
    ]
    return LinearSegmentedColormap.from_list("", palette)


def delay_cmap(vmin=-1, vmax=1):
    assert vmin <= 0 <= vmax
    zero = abs(vmin) / (abs(vmin) + vmax)
    palette = [
        (0, mcolors.colorConverter.to_rgba(colors["yellow"], alpha=1)),
        (zero, mcolors.colorConverter.to_rgba(colors["green"], alpha=1)),
        (
            zero + ((1 - zero) / 2.0),
            mcolors.colorConverter.to_rgba(colors["orange"], alpha=1),
        ),
        (1, mcolors.colorConverter.to_rgba(colors["red"], alpha=1)),
    ]
    return LinearSegmentedColormap.from_list("", palette)


def opaque_delay_cmap(vmin=-1, vmax=1):
    assert vmin < 0 < vmax
    zero = abs(vmin) / (abs(vmin) + vmax)
    palette = [
        (0, mcolors.colorConverter.to_rgba(colors["green"], alpha=0.01)),
        (zero, mcolors.colorConverter.to_rgba(colors["green"], alpha=0.01)),
        (zero + 0.1, mcolors.colorConverter.to_rgba(colors["yellow"], alpha=0.8)),
        (
            zero + 0.1 + ((1 - zero) / 2.0),
            mcolors.colorConverter.to_rgba(colors["orange"], alpha=1),
        ),
        (1, mcolors.colorConverter.to_rgba(colors["red"], alpha=1)),
    ]
    return LinearSegmentedColormap.from_list("", palette)
