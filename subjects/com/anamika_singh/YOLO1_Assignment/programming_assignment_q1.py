
def corner_to_center(x_min, y_min, x_max, y_max):
    """
    Convert bounding box from corner format (x_min, y_min, x_max, y_max)
    to center format (x_center, y_center, width, height).
    """
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    return x_center, y_center, width, height

def center_to_corner(x_center, y_center, width, height):
    """
    Convert bounding box from center format (x_center, y_center, width, height)
    to corner format (x_min, y_min, x_max, y_max).
    """
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return x_min, y_min, x_max, y_max


corner_box = (100, 150, 200, 250)
center_box = corner_to_center(*corner_box)
corner_box_converted_back = center_to_corner(*center_box)
print(f'Corner to center : {center_box}')
print(f'Center to corner : {corner_box_converted_back}')


