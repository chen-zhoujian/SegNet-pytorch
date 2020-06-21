import cv2 as cv
import numpy as np

paths = open("train.txt", "r")

CLASS_NUM = 2
SUM = [[] for i in range(CLASS_NUM)]
SUM_ = 0

for line in paths:
    line.rstrip("\n")
    line.lstrip("\n")
    path = line.split()
    img = cv.imread(path[1], 0)
    img_np = np.array(img)
    for i in range(CLASS_NUM):
        SUM[i].append(np.sum((img_np == i)))


for index, iter in enumerate(SUM):
    print("类别{}的数量：".format(index), sum(iter))


for iter in SUM:
    SUM_ += sum(iter)

median = 1/CLASS_NUM

for index, iter in enumerate(SUM):
    print("weight_{}:".format(index), median/(sum(iter)/SUM_))
