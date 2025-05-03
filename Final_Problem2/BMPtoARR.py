import numpy as np
from PIL import Image
from mpmath.libmp import to_int
from sympy.codegen.ast import int64, uint8

np.set_printoptions(threshold=np.inf)
bw_img1 = np.array(Image.open("./maps/map1.bmp").convert('1')).T
bw_img2 = np.array(Image.open("./maps/map2.bmp").convert('1')).T

def checkscope(pos_x, pos_y, x_size, y_size, inputarray): #This function hopefully can be used to be iterated through the input array at given cords and return if that scope is all white or not
    #print(f"Checking ({pos_x}, {pos_y})")

    for x in range(pos_x, pos_x + x_size):
        for y in range(pos_y, pos_y + y_size):

            if (pos_x + x_size >= inputarray.shape[0]):
                #print(f"{pos_x+x_size} was too large to index")
                break
            if (pos_y + y_size >= inputarray.shape[1]):
                #print(f"{pos_y + y_size} was too large to index")
                break

            #print(f"inner array check ({x}, {y})")
            if inputarray[x][y] > 0:
                pass
            elif inputarray[x][y] == 0:
                return False
    return True


def compressbmp(inputarray, ab_sizex, ab_sizey): # if ab (x,y) = 11, then output arr will be (528/11, 532/11) = (48, 48)
    out_arr = np.full((int(inputarray.shape[0] / ab_sizex), int(inputarray.shape[1] / ab_sizey)),fill_value=1, dtype= np.uint8)
    #print(inputarray.shape) #(532,528)
    cordsinter_arr = np.full((int(inputarray.shape[0] / ab_sizex), int(inputarray.shape[1] / ab_sizey)), fill_value=0, dtype= object)

    for x in range(0,(inputarray.shape[0])): #0 - 527
        for y in range(0,(inputarray.shape[1])): #0 - 531
            X = int(x / ab_sizex)-1
            Y = int(y / ab_sizey)-1
            if x % ab_sizex == 0 and y % ab_sizey == 0:
                #print(f"{(X,Y)}->{(x,y)}")
                cordsinter_arr[X,Y] = (x,y)


    for x in range(0,int(inputarray.shape[0] / ab_sizex)): # 0-47
        for y in range(0,int(inputarray.shape[1] / ab_sizey)): #0-47
           # print(f"{cordsinter_arr[x,y][0]}, {cordsinter_arr[x,y][1]}")
            if checkscope(pos_x=cordsinter_arr[x,y][0], pos_y=cordsinter_arr[x,y][1], x_size=ab_sizex, y_size=ab_sizey, inputarray=inputarray):
                out_arr[x, y] = 255
            else:
                out_arr[x, y] = 0
    return out_arr

compressed_map1 = compressbmp(bw_img1,11,11)
compressed_map2 = compressbmp(bw_img2,11,11)

compressed_map1 = Image.fromarray(compressed_map1).convert('1')
compressed_map2 = Image.fromarray(compressed_map2).convert('1')

compressed_map1.save("./maps/map1_compressed(48,48).bmp")
compressed_map2.save("./maps/map2_compressed(48,48).bmp")
