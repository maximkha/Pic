import numpy as np
from numpy.lib.type_check import imag

def diffImg(image, offsetX, offsetY):
    W, H, C = image.shape
    #W = image.shape[0]
    #H = image.shape[1]
    
    #resp = lambda x: x/255 # x/(255**2) * 255
    #op = lambda x, y: x * y

    #resp = lambda x: (x**2)*.7
    #op = lambda x, y: x - y
    
    #resp = lambda x: x*256
    #op = lambda x, y: x / (y + 1)

    #resp = lambda x: x
    #op = lambda x, y: x % (y+1)

    resp = lambda x: x+(255/2)
    op = lambda x, y: (255-x)-(255-y)

    result = np.zeros_like(image)
    for i in range(W):
        for j in range(H):
            if 0 <= i + offsetX < W and  0 <= j + offsetY < H:
                result[i, j, 0] = min(255, max(0, resp(op(image[i, j, 0], image[i + offsetX, j + offsetY, 0]))))
                result[i, j, 1] = min(255, max(1, resp(op(image[i, j, 1], image[i + offsetX, j + offsetY, 1]))))
                result[i, j, 2] = min(255, max(2, resp(op(image[i, j, 2], image[i + offsetX, j + offsetY, 2]))))
            else:
                result[i, j, :] = 0
    return result

import numba
fastDiff = numba.jit(numba.int32[:,:,:](numba.int32[:,:,:], numba.int32, numba.int32))(diffImg)

from PIL import Image
imgArr = np.array(Image.open('WIN_20200729_14_44_11_Pro.jpg'))

output = fastDiff(imgArr.astype(np.int32), -1, 1).astype(np.uint8)

Image.fromarray(output).save("out.png")