#!/bin/python3
'''
This file contains the external function for the eMiLy.py code. Two examples
are provided below for testing the code. ExPolyFun() is a 2-variable sombrero
function which has exactly 30 (degenerate) local minima. ExTrigFun() is a
10-variable trigonometric function which, in the interval of [-1:1] for each
of its variables contains exactly 1024 (degenerate) local minima.

To use your own function, simply add your own function below and import
that into eMiLy.py instead of one of the examples.

'''

import numpy as np
twopi=8.0*np.arctan(1.0)

# simple 4th order anisotropic sombrero function f(x,y) with 30 local minima
def ExPolyFun(XX):
# the following sets a tolerance parameter used to avoid division by zero
    epsilon=1e-6
    if(XX.size!=2):
        print('ExFun encountered an error: Incorrect number of variables.')
        return
    EE=np.zeros(4)
    EE[0]=0.0
    EE[1]=-1.0
    EE[2]=5.0
    EE[3]=0.5
# The local minimum along x=0 (theta=pi/2) is determined by
# EE[0]+EE[1]*x**2+(EE[2]-EE[3])*x**2
# which happens when 0=2*EE[1]+4*(EE[2]-EE[3])*x**2
# which for the example parameters above reads 0=-1.0+9.0*x**2
# therefore x=sqrt(1/9.0)=0.33333333
#
# In gnuplot, this can be visualized as
# splot [-0.6:0.6][-0.6:0.6][-0.1:0]0-1*(x**2+y**2)+((x**2+y**2)**2)*(5+0.5*cos(10*atan(y/x)))
#
# Therefore, we can search the parameter space of [-1:1] for x and y to locate all
# 10 local minima, all of which will be 0.33333333 distance from the origin
    RR=np.linalg.norm(XX)
    if(abs(XX[0])<epsilon and abs(XX[1])<epsilon):
        value=EE[0]
    else:
        theta=np.arctan(XX[1]/(XX[0]+epsilon*XX[1]))
        value=EE[0]+EE[1]*(RR**2)+EE[2]*(RR**4)+EE[3]*np.cos(30*theta)*(RR**4)
    return value

# simple product of 10 independent cosines in a 10-dimensional parameter space
def ExTrigFun(XX):
    ndim=10 # number of dimensions
    if(XX.size!=ndim):
        print('ExFun encountered an error: Incorrect number of variables.')
        return
    costerms=np.zeros(ndim)
    costerms=2.0+np.cos(twopi*XX)
    value=np.prod(costerms)
# in the part of the parameter space where the absolute value of all components
# of XX is less than 1.0 there will be exactly 2**ndim local minima; all of
# them will satisfy abs(XX[ii])=0.5 for all values of ii. With the default
# ndim=10 this means 1024 local minima.
    return value
