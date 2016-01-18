# -*- coding: utf-8 -*-
"""
Write a function is_collide that detects if two circle intersect each other. In 2D space, we
only care about the circle’s position and its size. A cycle is represented by a tuple (x, y, r) where x, y are the
coordinates of cycle’s center point, and r is its radius. To detect collision, we need to compute the distance
between their centers, and check if the distance is less than or equal to the sum of their radius.
Input: two circle represented by two tuple
Output: boolean represents the detection result
"""


import math


def acceptCircles():
    global r1,r2,x1,x2,y1,y2
    x1= int(raw_input('Enter x1 :'))
    y1= int(raw_input('Enter y1 :'))
    r1= int(raw_input('Enter Radius :'))
    c1=(x1,y1,r1)
    
    
    x2= int(raw_input('Enter x2 :'))
    y2= int(raw_input('Enter y2 :'))
    r2= int(raw_input('Enter Radius :'))
    c2=(x2,y2,r2)
    
    
    
    print 'Tuple for circle c1'
    print c1
    print 'Tuple for circle c2'
    print c2
    
def is_collide():
    
    global r1,r2,x1,x2,y1,y2
    
    rad=r1+r2
    radsq=rad*rad
    
    xc=(x1-x2)
    xsq=math.pow(xc,2)
    
    yc=(y1-y2)
    ysq=math.pow(yc,2)
    
    c1c2=xsq+ysq
    print 'end of calculations'
    
    if radsq==c1c2:
        print('Circles touch each other')
        return False
    elif(radsq<c1c2):
        print('Circles do no collide')
        return False
    else:
        print('Passing True value to caller function')
        return True    
        
def main():
    acceptCircles()
    if is_collide():
        print'you have returned the value of a fuction successfully!'
        print'***Circles collide!****'
        
    
main()