
ARROW_LEN = 0.4
ARROW_COLOR = "black"
CENTER_OFFSET = 0.05


def draw_arrows(ax, policy, row, column):
    """ Draws arrows in cell """
    if policy == "U":
        draw_arrow_U(ax, row, column)
    elif policy == "D":
        draw_arrow_D(ax, row, column)
    elif policy == "L":
        draw_arrow_L(ax, row, column)
    elif policy == "R":
        draw_arrow_R(ax, row, column)
    elif policy == "UD":
        draw_arrow_U(ax, row, column)
        draw_arrow_D(ax, row, column)
    elif policy == "UL":
        draw_arrow_U(ax, row, column)
        draw_arrow_L(ax, row, column)
    elif policy == "UR":
        draw_arrow_U(ax, row, column)
        draw_arrow_R(ax, row, column)
    elif policy == "DL":
        draw_arrow_D(ax, row, column)
        draw_arrow_L(ax, row, column)
    elif policy == "DR":
        draw_arrow_D(ax, row, column)
        draw_arrow_R(ax, row, column)
    elif policy == "LR":
        draw_arrow_L(ax, row, column)
        draw_arrow_R(ax, row, column)
    elif policy == "UDL":
        draw_arrow_U(ax, row, column)
        draw_arrow_D(ax, row, column)
        draw_arrow_L(ax, row, column)
    elif policy == "UDR":
        draw_arrow_U(ax, row, column)
        draw_arrow_D(ax, row, column)
        draw_arrow_R(ax, row, column)
    elif policy == "ULR":
        draw_arrow_U(ax, row, column)
        draw_arrow_L(ax, row, column)
        draw_arrow_R(ax, row, column)
    elif policy == "DLR":
        draw_arrow_D(ax, row, column)
        draw_arrow_L(ax, row, column)
        draw_arrow_R(ax, row, column)
    else:
        draw_arrow_U(ax, row, column)
        draw_arrow_D(ax, row, column)
        draw_arrow_L(ax, row, column)
        draw_arrow_R(ax, row, column)


def draw_arrow_U(ax, row, column):
    """ Draws arrow pointing to north """
    ax.annotate("", xy=(row, column - ARROW_LEN), xytext=(row, column + CENTER_OFFSET), arrowprops=dict(arrowstyle="->", color=ARROW_COLOR))


def draw_arrow_D(ax, row, column):
    """ Draws arrow pointing to south """
    ax.annotate("", xy=(row, column + ARROW_LEN), xytext=(row, column - CENTER_OFFSET), arrowprops=dict(arrowstyle="->", color=ARROW_COLOR))


def draw_arrow_L(ax, row, column):
    """ Draws arrow pointing to west """
    ax.annotate("", xy=(row - ARROW_LEN, column), xytext=(row + CENTER_OFFSET, column), arrowprops=dict(arrowstyle="->", color=ARROW_COLOR))


def draw_arrow_R(ax, row, column):
    """ Draws arrow pointing to east """
    ax.annotate("", xy=(row + ARROW_LEN, column), xytext=(row - CENTER_OFFSET, column), arrowprops=dict(arrowstyle="->", color=ARROW_COLOR))
