import numpy as np
from PIL import Image
from mpmath.libmp import to_int
from sympy.codegen.ast import int64

map_1 = Image.open("./maps/map1.bmp") # 532 x 528
map_2 = Image.open("./maps/map2.bmp") # 532 x 528
bw_img1 = map_1.convert('1')  # Convert to 1-bit black & white
bw_img2 = map_2.convert('1')  # Convert to 1-bit black & white

np.set_printoptions(threshold=np.inf)

bw_img1 = np.array(map_1)
bw_img2 = np.array(map_2)

def checkscope(pos_x, pos_y, x_size, y_size, inputarray): #This function hopefully can be used to be iterated through the input array at given cords and return if that scope is all white or not
    for x in range(pos_x, pos_x + x_size):
        for y in range(pos_y, pos_y + y_size):
            if inputarray[x][y] == 1:
                pass
            if inputarray[x][y] == 0:
                return False
    return True


def compressbmp(inputarray, ab_sizex, ab_sizey):
    out_arr = np.zeros((int(inputarray.shape[0]/ab_sizex), int(inputarray.shape[1]/ab_sizey)), dtype=int)
    for (x) in range(out_arr.shape[0]):
        for (y) in range(out_arr.shape[1]):

            if x % ab_sizex == 0 and y % ab_sizey == 0:
                if checkscope(pos_x=x, pos_y=y, x_size=ab_sizex, y_size=ab_sizey, inputarray=inputarray) == True:
                    out_arr[x,y] = 1
                else:
                    out_arr[x,y] = 0

    return out_arr

compressed_map1 = compressbmp(bw_img1,11,11)
compressed_map2 = compressbmp(bw_img2,11,11)
print(bw_img1.shape)
print(compressed_map1.shape)
print(bw_img1)
print(compressed_map1)

compressed_map1 = Image.fromarray(compressed_map1)
compressed_map2 = Image.fromarray(compressed_map2)

compressed_map1.save("map1_compressed(48,48).bmp")
compressed_map2.save("map2_compressed(48,48).bmp")