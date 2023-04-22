import math 
from timeit import default_timer as timer
from itertools import product

BACKGROUND = 255
FOREGROUND = 0

def secure_binary(image):
    width, height = image.shape
    for y, x in product(range(height), range(width)):
        if image[x, y] < 127:
            image[x, y] = FOREGROUND
        else:
            image[x, y] = BACKGROUND
    return image

def run_hol(image):
    image = secure_binary(image)
    n = image.shape[1]

    start_time = timer()

    def get(x, y):
        if (0 <= x and x < n) and (0 <= y and y < n):
            return image[y][x]
        return math.inf

    def label(x, y):
        while x >= 1 and get(x - 1, y) == FOREGROUND:
            x = x - 1    
        m = x
        while m < n and get(m, y) == FOREGROUND:
            image[y][m] = l
            m = m + 1
        
        x = x - 1
        while x <= m:
            if get(x, y - 1) == FOREGROUND:
                label(x, y - 1)
            if get(x, y + 1) == FOREGROUND:
                label(x, y + 1)
            x = x + 1

    l = 1
    for i in range (len(image)):
        for j in range (len(image)):
            if get(i, j) == FOREGROUND:
                label(i, j)
                l = l + 1
    
    end_time = timer()
    return image, end_time - start_time