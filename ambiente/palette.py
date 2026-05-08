"""ARC-AGI default 10-color palette.

Colors are stored as hex strings; index = color id used inside Grid tensors.
Override per-Grid by passing `palette=...` to the Grid constructor.
"""

ARC_PALETTE = (
    "#000000",  # 0  black     (background)
    "#0074D9",  # 1  blue
    "#FF4136",  # 2  red
    "#2ECC40",  # 3  green
    "#FFDC00",  # 4  yellow
    "#AAAAAA",  # 5  grey
    "#F012BE",  # 6  magenta
    "#FF851B",  # 7  orange
    "#7FDBFF",  # 8  light-blue
    "#870C25",  # 9  maroon
)

DEFAULT_PALETTE = ARC_PALETTE
DEFAULT_BACKGROUND = 0
N_COLORS = len(ARC_PALETTE)
