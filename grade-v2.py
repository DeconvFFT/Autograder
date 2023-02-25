#!/usr/bin/env python
# coding: utf-8

# In[287]:


from PIL import Image
from PIL import ImageFilter
import sys
import random

im = Image.open("test-images/a-3.jpg")

# Check its width, height, and number of color channels
print("Image is %s pixels wide." % im.width)
print("Image is %s pixels high." % im.height)
print("Image mode is %s." % im.mode)

im = im.filter(ImageFilter.GaussianBlur)
im.thumbnail((475,550), Image.ANTIALIAS)


# In[288]:


import math as math
import random
import numpy as np

def pixelValues(coordinateArray):
    pvs = [im.getpixel(coordinate) for coordinate in coordinateArray]
    return pvs

k = 2
clusterCenters = [random.randint(0,255) for _ in range(k)]
clusterAssignments = {i:[] for i in range(k)}
print("Initial Cluster Centers: ", clusterCenters)
for x in range(50):
    clusterAssignments = {i:[] for i in range(k)}
    for x in range(im.width):
        for y in range(im.height):
            p = im.getpixel((x,y))
            closestCenterIndex,closestCenterValue = -1,999
            for cluster_index, cluster_value in enumerate(clusterCenters):
                distance = abs(p-cluster_value)
                if distance < closestCenterValue:
                    closestCenterValue = distance
                    closestCenterIndex = cluster_index
            clusterAssignments[closestCenterIndex].append((x,y))

    clustersChanged = False
    for cluster in clusterAssignments:
       pvs = pixelValues(clusterAssignments[cluster])
       newCenter = int(np.mean(pvs))
       if newCenter != clusterCenters[cluster]:
           clustersChanged=True
       clusterCenters[cluster] = newCenter

    print("New Cluster Centers: ", clusterCenters)
    if clustersChanged is False:
        break


# In[289]:


blacks = clusterAssignments[np.argmin(clusterCenters)]
whites = clusterAssignments[np.argmax(clusterCenters)]

for pixel in blacks:
    im.putpixel((pixel),0)
for pixel in whites:
    im.putpixel((pixel),255)

im.save("segmented.jpg")


# In[290]:


#Find all black regions


answersDetected = 0
percentageThreshold = 85
im_highlighted = im.copy()
im_highlighted = im_highlighted.convert("RGB")
square_size = 8
y = im.height//5
x = im.width//9
step_size_y = 1
step_size_x = 1

while y < im.height-10:
    while x < im.width-im.width//9:
        step_size_x = 1
        numBlacks = 0
        totalRegion = square_size * square_size
        for x_offset in range(square_size):
            for y_offset in range(square_size):
                numBlacks += int(im.getpixel((x+x_offset,y+y_offset)) ==0)
        percentageBlack = numBlacks/totalRegion * 100
        if percentageBlack >= percentageThreshold:
            step_size_x = 10
            step_size_y = 1
            answersDetected +=1
            for x_offset in range(square_size):
                for y_offset in range(square_size):
                    im_highlighted.putpixel((x+x_offset,y+y_offset), (255,0,0))
        x = x+step_size_x

    x = im.width//9
    y = y+step_size_y


print("At threshold percent", percentageThreshold, ":", answersDetected, "answers were detected")

im_colored_template = im_highlighted.copy()
im_highlighted.save("colored.jpg")


# In[291]:


# find y coordinate of first answer box
answersDetected = 0
percentageThreshold = 30
im_highlighted = im.copy()
im_highlighted = im_highlighted.convert("RGB")
rectangle_width= 300
rectangle_height = 10
y = im.height//5
x = im.width//9
step_size_y = 1
step_size_x = 1
terminate = False
while x < im.width-im.width//9:
    while y < im.height-10:
        step_size_x = 1
        numBlacks = 0
        totalRegion = rectangle_width * rectangle_height
        for x_offset in range(rectangle_width):
            for y_offset in range(rectangle_height):
                numBlacks += int(im.getpixel((x+x_offset,y+y_offset)) ==0)
        percentageBlack = numBlacks/totalRegion * 100
        if percentageBlack >= percentageThreshold:
            step_size_x = 10
            step_size_y = 1
            answersDetected +=1
            terminate = True
            break
        y = y+step_size_y
    if terminate is True:
        break
    y = im.height//5
    x = x+step_size_x


print("At threshold percent", percentageThreshold, ":", answersDetected, "answers were detected")

# find x coordinate of first answer box
percentageThreshold = 30
im_highlighted = im.copy()
im_highlighted = im_highlighted.convert("RGB")
rectangle_width= 10
rectangle_height = 3
x = im.width//9
newY = y
y= newY
step_size_y = 1
step_size_x = 1
terminate = False
while y < im.height-10:
    while x < im.width-im.width//9:
        step_size_x = 1
        numBlacks = 0
        totalRegion = rectangle_width * rectangle_height
        for x_offset in range(rectangle_width):
            for y_offset in range(rectangle_height):
                numBlacks += int(im.getpixel((x+x_offset,y+y_offset)) ==0)
        percentageBlack = numBlacks/totalRegion * 100
        if percentageBlack >= percentageThreshold:
            step_size_x = 10
            step_size_y = 1
            answersDetected +=1
            for x_offset in range(rectangle_width):
                for y_offset in range(rectangle_height):
                    im_highlighted.putpixel((x+x_offset,y+y_offset), (255,0,0))

            terminate = True
            break
        x = x+step_size_x
    if terminate is True:
        break
    y = y +step_size_y



im_highlighted.putpixel((x+5,y),(0,0,255))
im_highlighted.putpixel((x+5,y-1),(0,0,255))
im_highlighted.putpixel((x+5,y-2),(0,0,255))
im_highlighted.putpixel((x+5,y+1),(0,0,255))
im_highlighted.putpixel((x+5,y+2),(0,0,255))
im_highlighted.putpixel((x+4,y),(0,0,255))
im_highlighted.putpixel((x+4,y-1),(0,0,255))
im_highlighted.putpixel((x+4,y-2),(0,0,255))
im_highlighted.putpixel((x+4,y+1),(0,0,255))
im_highlighted.putpixel((x+4,y+2),(0,0,255))

print("Corner Coordinate:",(x,y))

START_COORDINATE = (x+5,y+5)

im_highlighted.save("corner.jpg")


# In[299]:


x = START_COORDINATE[0]
y = START_COORDINATE[1]

im_colored = im_colored_template.copy()
#starting at center of first box (start_coordinate), jump right RIGHT_JUMP pixels at a time, reading whether
#there is a red pixel present, then reset x to start coordinate and jump down DOWN_JUMP pixels


RIGHT_JUMP = 15
DOWN_JUMP = 12
question_number = 1
answerDictionary = {i:[] for i in range(1,86)}
allWhite = True
while True:
    y_offset = DOWN_JUMP * (question_number - 1)
    allWhite = True
    for x_offset,symbol in zip([RIGHT_JUMP * i for i in range(5)], ["A","B","C","D","E"]):

        p = im_colored.getpixel((x+x_offset, y+y_offset))
       

        for i in range(-3,1):
            im_colored.putpixel((x+x_offset, y+y_offset-i), (0,255,0))
        if p == (255,0,0):
            answerDictionary[question_number].append(symbol)
            allWhite = False
        if p == (0,0,0):
            allWhite = False

    margin_x = START_COORDINATE[0] - int(RIGHT_JUMP * 1.5) if question_number <10 else START_COORDINATE[0] - int(RIGHT_JUMP * 1.75)
    numBlacks = 0
    rectangle_height = 10
    rectangle_width = 10
    totalRegion = rectangle_width * rectangle_height
    percentageThreshold = 10
    for x_kernel in range(rectangle_width):
        for y_kernel in range(rectangle_height):
            numBlacks += int(im_colored.getpixel((margin_x+x_kernel,y+y_offset-5+y_kernel)) ==0)
            im_colored.putpixel((margin_x+x_kernel,y+y_offset-5+y_kernel), (0,255,0))
    percentageBlack = numBlacks/totalRegion * 100
    if percentageBlack >= percentageThreshold:
       answerDictionary[question_number].append('x')
    if not allWhite:
        question_number+=1
    else:
        break




im_colored.save("colored2.jpg")


print("Answers Detected Column 1:", answerDictionary)



#find next column,update X part of START_COORDINATE, then repeat above procedure
x = START_COORDINATE[0] + (RIGHT_JUMP *6)




#find next column, update X part of START_COORDINATE, then repeat above procedure








# In[292]:




