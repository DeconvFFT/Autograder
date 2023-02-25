from PIL import Image as im
import numpy as np
from PIL import ImageDraw, ImageFilter
import sys
def convertInt(str):
    value = 0
    for c in str:
        if("A" == c):
            mask = 1 << 4
            value = mask ^ value
        if("B" == c):
            mask = 1 << 3
            value = mask ^ value
        if("C" == c):
            mask = 1 << 2
            value = mask ^ value
        if("D" == c):
            mask = 1 << 1
            value = mask ^ value
        if("E" == c):
            value = 1 ^ value
    return value

def intToPixel(num):
    pixels = []
    bits = 0
    
    while (num):
        if num & 1 == 1:
            pixels.insert(0, 255)
        else:
            pixels.insert(0, 0)
        num >>= 1
        bits += 1
    
    while bits < 5:
        pixels.insert(0, 0)
        bits += 1
    return pixels

def answerToBarcode(fileName):
    pixels = []
    file1 = open(fileName,'r')
    lines = file1.readlines()
    for line in lines:
        answers = [x for x in line.split()[1]]
        pixels.extend(intToPixel(convertInt(answers)))
    return pixels


if __name__ == "__main__":
    print(sys.argv)
    form_im=im.open(sys.argv[1])
    pixel_arr=answerToBarcode(sys.argv[2])
    trailing=[0]*10
    pixels=trailing+pixel_arr+trailing
    print(pixels)
    t=np.asarray(pixels)
    print(t)
    dup=np.repeat([np.repeat(t,2)],repeats=50,axis=0)
    print(dup)
    barcode_im = im.fromarray(np.uint8(dup),'L')
    form_im.paste(barcode_im,(40, 30))
    form_im.save(sys.argv[3])