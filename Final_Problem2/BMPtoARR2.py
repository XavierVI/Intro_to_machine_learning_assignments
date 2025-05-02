import numpy as np
from PIL import Image


def compress_map(img_path, ab_sizex, ab_sizey):
    """
    Compresses a BMP map image using a pooling-like approach.

    Args:
        img_path (str): Path to the BMP image.
        ab_sizex (int): Block size in the x-direction.
        ab_sizey (int): Block size in the y-direction.

    Returns:
        np.ndarray: The compressed map as a NumPy array (0: obstacle, 1: free space).
    """

    img = Image.open(img_path).convert('1')  # Open and convert to 1-bit B/W
    # Convert to NumPy array of ints (0 or 255)
    img_array = np.array(img).astype(int)
    img_array = img_array // 255  # Normalize to 0 and 1 (0: black, 1: white)

    original_height, original_width = img_array.shape
    compressed_height = (original_height + ab_sizex -
                         1) // ab_sizex  # Ceiling division
    compressed_width = (original_width + ab_sizey - 1) // ab_sizey

    compressed_map = np.zeros((compressed_height, compressed_width), dtype=int)

    for i in range(compressed_height):
        for j in range(compressed_width):
            start_x = i * ab_sizex
            end_x = min(start_x + ab_sizex, original_height)
            start_y = j * ab_sizey
            end_y = min(start_y + ab_sizey, original_width)

            block = img_array[start_x:end_x, start_y:end_y]
            compressed_map[i, j] = 1 if np.all(block == 1) else 0

    return compressed_map


# Example Usage
map_1_path = "./maps/map1.bmp"
map_2_path = "./maps/map2.bmp"
ab_sizex = 11
ab_sizey = 11

compressed_map_1_array = compress_map(map_1_path, ab_sizex, ab_sizey)
compressed_map_2_array = compress_map(map_2_path, ab_sizex, ab_sizey)

print("Compressed Map 1 Shape:", compressed_map_1_array.shape)
print("Compressed Map 2 Shape:", compressed_map_2_array.shape)

# If you need to save as BMP (not required by the task, but for visualization):


def array_to_image(arr):
    img_array = (arr * 255).astype(np.uint8)
    return Image.fromarray(img_array, mode='L')  # 'L' mode for grayscale


array_to_image(compressed_map_1_array).save("map1_compressed.bmp")
array_to_image(compressed_map_2_array).save("map2_compressed.bmp")
