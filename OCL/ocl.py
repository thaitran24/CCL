import cv2
import random
import numpy as np
from PIL import Image
from timeit import default_timer as timer
from itertools import product
from union_find import UnionFindArray

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

def run_ocl_pil(image):
    width, height = image.size

    data = image.load()
    out_image = Image.new('RGB', (width, height))
    outdata = out_image.load()

    uf = UnionFindArray()
    labels = {}

    start_time = timer()

    """FIRST PASS"""
    for y, x in product(range(height), range(width)):
        """
        Forward scan mask:
            -------------
            | a | b | c |
            -------------
            | d | e |   |
            -------------
        The current position is e.
        The scanned position is a, b, c, d.
        The mask travel left to right, top to bottom.
        This first pass follow the first decision tree from paper.
        """
 
        if data[x, y] == BACKGROUND:
            """Ignore background pixel"""
            pass
 
        elif y > 0 and data[x, y - 1] == FOREGROUND:
            """
            If pixel in b is background, so pixel a, d and c are its neighbors, 
            so they are all part of the same component. Don't need to check their labels.
            """
            labels[x, y] = labels[(x, y - 1)]

        elif x+1 < width and y > 0 and data[x + 1, y - 1] == 0:
            """
            If c is foreground, b is its neighbor, but a and d are not. 
            Therefore, we must check a and d's labels.
            """
            c = labels[(x + 1, y - 1)]
            labels[x, y] = c
 
            if x > 0 and data[x - 1, y - 1] == 0:
                """
                If a is foreground, then a and c are connected through e. 
                Therefore, we must union their sets.
                """
                a = labels[(x - 1, y - 1)]
                uf.union(c, a)
 
            elif x > 0 and data[x - 1, y] == 0:
                """
                If d is foreground, then d and c are connected through e.
                Therefore we must union their sets.
                """
                d = labels[(x - 1, y)]
                uf.union(c, d)
 
        elif x > 0 and y > 0 and data[x - 1, y - 1] == 0:
            """
            If a is foreground (we already have b and c are backgrounds),
            d is a's neighbor, so they already have the same label, then assign a's label to e.
            """
            labels[x, y] = labels[(x - 1, y - 1)]
 
        elif x > 0 and data[x - 1, y] == 0:
            """
            If d is foreground (we already have a, b, c are backgrounds), 
            then assign d's label to e.
            """
            labels[x, y] = labels[(x - 1, y)]

        else:
            """
            All neighbors are background, then create new label.
            """ 
            labels[x, y] = uf.make_label()
 
    """SECOND PASS"""
    uf.flatten()
    colors = {}

    for (x, y) in labels:
 
        # Name of the component the current point belongs to
        component = uf.find(labels[(x, y)])

        # Update the labels with correct information
        labels[(x, y)] = component
 
        # Associate a random color with this component 
        if component not in colors: 
            colors[component] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Colorize the image
        outdata[x, y] = colors[component]

    end_time = timer()
    
    return (labels, out_image, end_time - start_time)


def run_ocl_np(image):
    image = secure_binary(image)
    width, height = image.shape
    out_image = np.zeros((width, height, 3))

    uf = UnionFindArray()
    labels = {}

    start_time = timer()

    """FIRST PASS"""
    for y, x in product(range(height), range(width)):
        """
        Forward scan mask:
            -------------
            | a | b | c |
            -------------
            | d | e |   |
            -------------
        The current position is e.
        The scanned position is a, b, c, d.
        The mask travel left to right, top to bottom.
        This first pass follow the first decision tree from paper.
        """
 
        if image[x, y] == BACKGROUND:
            """Ignore background pixel"""
            pass
 
        elif y > 0 and image[x, y - 1] == FOREGROUND:
            """
            If pixel in b is background, so pixel a, d and c are its neighbors, 
            so they are all part of the same component. Don't need to check their labels.
            """
            labels[x, y] = labels[(x, y - 1)]

        elif x+1 < width and y > 0 and image[x + 1, y - 1] == 0:
            """
            If c is foreground, b is its neighbor, but a and d are not. 
            Therefore, we must check a and d's labels.
            """
            c = labels[(x + 1, y - 1)]
            labels[x, y] = c
 
            if x > 0 and image[x - 1, y - 1] == 0:
                """
                If a is foreground, then a and c are connected through e. 
                Therefore, we must union their sets.
                """
                a = labels[(x - 1, y - 1)]
                uf.union(c, a)
 
            elif x > 0 and image[x - 1, y] == 0:
                """
                If d is foreground, then d and c are connected through e.
                Therefore we must union their sets.
                """
                d = labels[(x - 1, y)]
                uf.union(c, d)
 
        elif x > 0 and y > 0 and image[x - 1, y - 1] == 0:
            """
            If a is foreground (we already have b and c are backgrounds),
            d is a's neighbor, so they already have the same label, then assign a's label to e.
            """
            labels[x, y] = labels[(x - 1, y - 1)]
 
        elif x > 0 and image[x - 1, y] == 0:
            """
            If d is foreground (we already have a, b, c are backgrounds), 
            then assign d's label to e.
            """
            labels[x, y] = labels[(x - 1, y)]

        else:
            """
            All neighbors are background, then create new label.
            """ 
            labels[x, y] = uf.make_label()
 
    """SECOND PASS"""
    uf.flatten()
    colors = {}

    for (x, y) in labels:
 
        # Name of the component the current point belongs to
        component = uf.find(labels[(x, y)])

        # Update the labels with correct information
        labels[(x, y)] = component
 
        # Associate a random color with this component 
        if component not in colors: 
            colors[component] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Colorize the image
        out_image[x, y] = colors[component]

    end_time = timer()
    return (labels, out_image, end_time - start_time)


def save_image(image, vis_dir, idx):
    cv2.imwrite('{}/{:010d}.jpg'.format(vis_dir, idx), image)
    return idx + 1

def apply_color(image, x, y, color, vis_dir, idx):
    image[x, y] = color
    return save_image(image, vis_dir, idx)

def vis_run_ocl(image, vis_dir):
    image = secure_binary(image)

    width, height = image.shape
    out_image = np.zeros((width, height, 3))

    for y, x in product(range(height), range(width)):
        if image[x, y] == FOREGROUND:
            out_image[x, y] = (FOREGROUND, FOREGROUND, FOREGROUND)
        else:
            out_image[x, y] = (BACKGROUND, BACKGROUND, BACKGROUND)

    uf = UnionFindArray()
    labels = {}

    start_time = timer()

    mask_color = (192,192,192)
    label_colors = [
        (255,0,0),
        (0,255,0),
        (0,0,255),
        (255,255,0),
        (0,255,255),
        (255,0,255),
        (128,128,128),
        (128,0,0),
        (128,128,0),
        (0,128,0),
        (128,0,128),
        (0,128,128),
        (0,0,128)
    ]

    idx = 0
    """FIRST PASS"""
    for x, y in product(range(width), range(height)):
        if image[x, y] == BACKGROUND:
            pass
 
        elif y > 0 and image[x, y - 1] == FOREGROUND:
            tmp_color = out_image[x, y - 1]
            idx = apply_color(out_image, x, y - 1, mask_color, vis_dir, idx)

            labels[x, y] = labels[(x, y - 1)]
            
            idx = apply_color(out_image, x, y, label_colors[labels[x, y]], vis_dir, idx)
            idx = apply_color(out_image, x, y - 1, tmp_color, vis_dir, idx)

        elif x + 1 < width and y > 0 and image[x + 1, y - 1] == FOREGROUND:
            tmp_color = out_image[x + 1, y - 1]
            idx = apply_color(out_image, x + 1, y - 1, mask_color, vis_dir, idx)

            c = labels[(x + 1, y - 1)]
            labels[x, y] = c

            idx = apply_color(out_image, x, y, label_colors[c], vis_dir, idx)
            idx = apply_color(out_image, x + 1, y - 1, tmp_color, vis_dir, idx)

            if x > 0 and image[x - 1, y - 1] == FOREGROUND:
                tmp_color = out_image[x - 1, y - 1]
                idx = apply_color(out_image, x - 1, y - 1, mask_color, vis_dir, idx)

                a = labels[(x - 1, y - 1)]
                uf.union(c, a)

                idx = apply_color(out_image, x - 1, y - 1, tmp_color, vis_dir, idx)
 
            elif x > 0 and image[x - 1, y] == FOREGROUND:
                tmp_color = out_image[x - 1, y]
                idx = apply_color(out_image, x - 1, y, mask_color, vis_dir, idx)

                d = labels[(x - 1, y)]
                uf.union(c, d)

                idx = apply_color(out_image, x - 1, y, tmp_color, vis_dir, idx)
 
        elif x > 0 and y > 0 and image[x - 1, y - 1] == FOREGROUND:
            tmp_color = out_image[x - 1, y - 1]
            idx = apply_color(out_image, x - 1, y - 1, mask_color, vis_dir, idx)

            labels[x, y] = labels[(x - 1, y - 1)]

            idx = apply_color(out_image, x, y, label_colors[labels[x, y]], vis_dir, idx)
            idx = apply_color(out_image, x - 1, y - 1, tmp_color, vis_dir, idx)
 
        elif x > 0 and image[x - 1, y] == FOREGROUND:
            tmp_color = out_image[x - 1, y]
            idx = apply_color(out_image, x - 1, y, mask_color, vis_dir, idx)

            labels[x, y] = labels[(x - 1, y)]

            idx = apply_color(out_image, x, y, label_colors[labels[x, y]], vis_dir, idx)
            idx = apply_color(out_image, x - 1, y, tmp_color, vis_dir, idx)

        else:
            labels[x, y] = uf.make_label()
            idx = apply_color(out_image, x, y, label_colors[labels[x, y]], vis_dir, idx)
 
    """SECOND PASS"""
    uf.flatten()
    colors = {}

    for (x, y) in labels:
        component = uf.find(labels[(x, y)])
        labels[(x, y)] = component
        if component not in colors: 
            colors[component] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        out_image[x, y] = colors[component]
        idx = apply_color(out_image, x, y, colors[component], vis_dir, idx)

    end_time = timer()
    return (labels, out_image, end_time - start_time)