import math


def find_equidistant_point(x1, y1, x2, y2):
    ax = x2 - x1
    ay = y2 - y1

    nx = 1
    ny = -ax / ay

    nx = nx / math.hypot(nx, ny)
    ny = ny / math.hypot(nx, ny)

    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2

    k = math.sqrt(3) * math.dist((x1, y1), (x2, y2)) / 2

    x3 = xc + k * nx
    y3 = yc + k * ny

    return x3, y3


def find_circle(x1, y1, x2, y2, x3, y3):
    # https://stackoverflow.com/a/50974391/20380842
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = x2 * x2 + y2 * y2
    bc = (x1 * x1 + y1 * y1 - temp) / 2
    cd = (temp - x3 * x3 - y3 * y3) / 2
    det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2)

    if abs(det) < 1.0e-6:
        return (None, float('inf'))

    # Center of circle
    cx = (bc * (y2 - y3) - cd * (y1 - y2)) / det
    cy = ((x1 - x2) * cd - (x2 - x3) * bc) / det

    radius = math.sqrt((cx - x1) ** 2 + (cy - y1) ** 2)
    return cx, cy, radius