#!/usr/bin/python3
import random
import os 

if __name__ == '__main__':
    r = random.random()
    if r < 0.5:
        print('Vulkan first')
        os.system('./vulkanrs.out')
        os.system('./cudac.out')
    else:
        print('Cuda first')
        os.system('./cudac.out')
        os.system('./vulkanrs.out')