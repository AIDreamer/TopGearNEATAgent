import ctypes

road_color1 = (41, 66, 86)
road_color2 = (66, 82, 82)
road_color3 = (206, 206, 206)
road_color4 = (156, 0, 0)

def convert_rgb_to_int(rgb):
    r, g, b = rgb
    return (b << 24) + (g << 16) + (r << 8) + 255;

print(ctypes.c_int32(convert_rgb_to_int(road_color1)).value)
print(ctypes.c_int32(convert_rgb_to_int(road_color2)).value)
print(ctypes.c_int32(convert_rgb_to_int(road_color3)).value)
print(ctypes.c_int32(convert_rgb_to_int(road_color4)).value)