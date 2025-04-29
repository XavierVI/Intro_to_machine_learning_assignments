import numpy as np
from PIL import Image

map_1 = Image.open("\maps\map1.bmp") # 532 x 528
map_2 = Image.open("\maps\map2.bmp") # 532 x 528

map0_arr = np.array(map_1)
map1_arr = np.array(map_2)

