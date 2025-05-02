import numpy as np
from PIL import Image
from mpmath.libmp import to_int
from sympy.codegen.ast import int64, uint8

map_1 = Image.open("./maps/map1.bmp")  # 532 x 528
map_2 = Image.open("./maps/map2.bmp")  # 532 x 528
bw_img1 = map_1.convert('1')  # Convert to 1-bit black & white
bw_img2 = map_2.convert('1')  # Convert to 1-bit black & white

np.set_printoptions(threshold=np.inf)

bw_img1 = np.array(map_1)
bw_img2 = np.array(map_2)


# This function hopefully can be used to be iterated through the input array at given cords and return if that scope is all white or not
def checkscope(pos_x, pos_y, x_size, y_size, inputarray):
   # print(f"Checking ({pos_x}, {pos_y})")
    for x in range(pos_x, pos_x + x_size-1):
        for y in range(pos_y, pos_y + y_size-1):
            # print(f"inner array check ({x}, {y})")
            x -= 1
            y -= 1
            if inputarray[x][y] == 1:
                pass
            elif inputarray[x][y] == 0:
                return False
            else:
                return True
    return False


# if ab (x,y) = 11, then output arr will be (528/11, 532/11) = (48, 48)
def compressbmp(inputarray, ab_sizex, ab_sizey):
    out_arr = np.full((int(inputarray.shape[0] / ab_sizex), int(
        inputarray.shape[1] / ab_sizey)), fill_value=-1, dtype=np.int16)
    # print(inputarray.shape) #(528,532)
    cordsinter_arr = np.full((int(inputarray.shape[0] / ab_sizex), int(
        inputarray.shape[1] / ab_sizey)), fill_value=0, dtype=object)

    for x in range(0, inputarray.shape[0]-1):  # 0 - 528
        for y in range(0, inputarray.shape[1]-1):  # 0 - 532
            X = int(x / ab_sizex) - 1
            Y = int(y / ab_sizey) - 1
            if x % ab_sizex == 0 and y % ab_sizey == 0:
                cordsinter_arr[X, Y] = (x, y)

    for x in range(0, int(inputarray.shape[0] / ab_sizex)):
        for y in range(0, int(inputarray.shape[1] / ab_sizey)):
           # print(f"{cordsinter_arr[x,y][0]}, {cordsinter_arr[x,y][1]}")
            if checkscope(pos_x=cordsinter_arr[x, y][0], pos_y=cordsinter_arr[x, y][1], x_size=ab_sizex, y_size=ab_sizey, inputarray=inputarray):
                out_arr[x, y] = 255
            else:
                out_arr[x, y] = 0
    return out_arr


compressed_map1 = compressbmp(bw_img1, 11, 11)
compressed_map2 = compressbmp(bw_img2, 11, 11)

compressed_map1 = Image.fromarray(compressed_map1)
compressed_map2 = Image.fromarray(compressed_map2)

compressed_map1.convert("L").save("./maps/map1_compressed(48,48).bmp")
compressed_map2.convert("L").save("./maps/map2_compressed(48,48).bmp")
