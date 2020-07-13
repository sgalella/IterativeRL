
ARROW_LEN = 0.3
ARROW_COLOR = "black"
CENTER_OFFSET = 0.05


def draw_arrows(max_neighbors, ax, row, column):
    """ Draws arrows in cell """
    if max_neighbors == [1, 0, 0, 0]:
        draw_arrow_N(ax, row, column)
    elif max_neighbors == [0, 1, 0, 0]:
        draw_arrow_S(ax, row, column)
    elif max_neighbors == [0, 0, 1, 0]:
        draw_arrow_W(ax, row, column)
    elif max_neighbors == [0, 0, 0, 1]:
        draw_arrow_E(ax, row, column)
    elif max_neighbors == [1, 1, 0, 0]:
        draw_arrow_N(ax, row, column)
        draw_arrow_S(ax, row, column)
    elif max_neighbors == [1, 0, 1, 0]:
        draw_arrow_N(ax, row, column)
        draw_arrow_W(ax, row, column)
    elif max_neighbors == [1, 0, 0, 1]:
        draw_arrow_N(ax, row, column)
        draw_arrow_E(ax, row, column)
    elif max_neighbors == [0, 1, 1, 0]:
        draw_arrow_S(ax, row, column)
        draw_arrow_W(ax, row, column)
    elif max_neighbors == [0, 1, 0, 1]:
        draw_arrow_S(ax, row, column)
        draw_arrow_E(ax, row, column)
    elif max_neighbors == [0, 0, 1, 1]:
        draw_arrow_W(ax, row, column)
        draw_arrow_E(ax, row, column)
    elif max_neighbors == [1, 1, 1, 0]:
        draw_arrow_N(ax, row, column)
        draw_arrow_S(ax, row, column)
        draw_arrow_W(ax, row, column)
    elif max_neighbors == [1, 1, 0, 1]:
        draw_arrow_N(ax, row, column)
        draw_arrow_S(ax, row, column)
        draw_arrow_E(ax, row, column)
    elif max_neighbors == [1, 0, 1, 1]:
        draw_arrow_N(ax, row, column)
        draw_arrow_W(ax, row, column)
        draw_arrow_E(ax, row, column)
    elif max_neighbors == [0, 1, 1, 1]:
        draw_arrow_S(ax, row, column)
        draw_arrow_W(ax, row, column)
        draw_arrow_E(ax, row, column)
    else:
        draw_arrow_N(ax, row, column)
        draw_arrow_S(ax, row, column)
        draw_arrow_W(ax, row, column)
        draw_arrow_E(ax, row, column)


def draw_arrow_N(ax, row, column):
    """ Draws arrow pointing to north """
    ax.annotate("", xy=(row, column - ARROW_LEN), xytext=(row, column + CENTER_OFFSET), arrowprops=dict(arrowstyle="->", color=ARROW_COLOR))


def draw_arrow_S(ax, row, column):
    """ Draws arrow pointing to south """
    ax.annotate("", xy=(row, column + ARROW_LEN), xytext=(row, column - CENTER_OFFSET), arrowprops=dict(arrowstyle="->", color=ARROW_COLOR))


def draw_arrow_W(ax, row, column):
    """ Draws arrow pointing to west """
    ax.annotate("", xy=(row - ARROW_LEN, column), xytext=(row + CENTER_OFFSET, column), arrowprops=dict(arrowstyle="->", color=ARROW_COLOR))


def draw_arrow_E(ax, row, column):
    """ Draws arrow pointing to east """
    ax.annotate("", xy=(row + ARROW_LEN, column), xytext=(row - CENTER_OFFSET, column), arrowprops=dict(arrowstyle="->", color=ARROW_COLOR))
