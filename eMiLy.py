#!/bin/python3
'''
This is eMiLy version 1.0.3 (January 2019 release), a machine learning
minimum search program. Written by Viktor Zolyomi.

Requirements:
    Python 3
    libraries: numpy, scipy, random, json

Usage:

Select your choice of function for ExFun() on line 89 or 91, or import
a custom function of your own, then set the number of parameters and your
budget in main() according to your choice of function (see lines 261-266)

Description:

eMiLy.py is designed to combine active learning and random search for the
purpose of finding as many local minima of an externally provided
function ExFun() as possible, on a small budget. The code does so by treating
the minimum search as a classification problem where the local minima are the
classes and the points within the parameter space are classified according to
which local minimum they lead to by minimization. Search begins with a random
sweep which is then analyzed to identify the relevant regions of the parameter
space which should subsequently be scoured by active learning.

The function to be minimized must be provided in a separate python script. Two
simple example functions are provided in the file exfun.py. To use the code
with a different function, simply add the code of your custom formula to
exfun.py and import that function as ExFun() below instead of one of the
provided examples.

There are 2 core functionalities in this version of the code:

1) FullRand() performs a purely random search of the parameter space similar
to random forest except here a single decision tree is used. From each random
point a minimization is initiated using scipy.optimize.minimize. The function
outputs the found local minima and the random points from which they were
reached.

2) EffSphereSearch() takes as input the outcome of FullRand() and determines
whether the function has its minima scattered uniformly in the parameter space
or clustered in small parts of it. In the former case we expect to find more
local minima away from the ones already found, whereas in the latter we expect
the opposite. The code makes controlled random queries in the parts of the
parameter space where it expects to find new local minima.

The two example functions ExTrigFun() and ExPolyFun() are each an example
of a function with uniformly scattered and clustered minima, respectively.
Output from eMiLy.py for each is provided in separate files using a total
number of queries equal to the exact number of local minima in the parameter
space. The *.stdout files contain the standard output, while the *.LocMin.dat
files contain the local minima.

Planned updates for future versions:

EffSphereSearch() currently only handles functions with clustered and uniformly
scattered local minima. Some functions exhibit a behavior in between, i.e. they
might have clustered minima but lots of clusters which are scattered uniformly
all over the parameter space. For such functions, we must search both away from
and close to the effective attraction spheres. Support for this will be added
in the future. Moreover, the current threshold that chooses whether we expect
the minima of the function to be clustered or not is somewhat arbitrary and
needs optimization. Introduction of double precision is also planned.

Copyright:

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''

import json
import random
import numpy as np
from scipy.special import gamma as GammaFun
from scipy.optimize import minimize as spmin
# select the following to use the code with the 2-variable polynomial function in exfun.py 
#from exfun import ExPolyFun as ExFun
# select the following to use the code with the 10-variable cosine product function in exfun.py 
from exfun import ExTrigFun as ExFun

onepi=4.0*np.arctan(1.0) # value of constant pi

# here we set the distance tolerance for identifying duplicates of found local mimima;
# set this according to the needs of your external function, and if tight tolerance
# is required consider using double precision
locmintol=1e-4

# here we define a class that stores the local minima found by the code,
# keeping track of the data points from which they were reached
class LocMinArr:
    def __init__(self, LocMin, LocMinVal, MinOrig, LDup, DupPos, LMCount):
        self.LocMin = LocMin # the position of the local minimum
        self.LocMinVal = LocMinVal # function value at the local minimum
        self.MinOrig = MinOrig # the point from which we arrived at LocMin
        self.LDup = LDup # 0 or 1 stating whether the local minimum is a duplicate of one found before; set to -1 if minimization failed or 2 if it led us outside the parameter space
        self.DupPos = DupPos # integer index of the local minimum of which this is a duplicate (if LDup==true)
        self.LMCount = LMCount # the number of unique local minima

def FullRand(numdat,numpar,ttrange):
    '''
    FullRand() generates a random set of points on which to query ExFun and
    determines the nearest local minimum to each point.
    '''
    ttrand=np.zeros((numdat,numpar)) # random set of tt parameters for ExFun(tt)
    locmin=np.zeros((numdat,numpar)) # position of local minima in parameter space
    locminval=np.zeros((numdat)) # ExFun() values at local minima
    LDup=np.arange(0,numdat) # variable tracking whether the local minimum is unique (0), a duplicate (1), or if minimization failed (-1) or led outside the required range (2)
    DupPos=np.arange(0,numdat) # index of unique local minimum if the current local minimum is a duplicate; an offset of 1 is added for convenience
    LDup[:]=0
    DupPos[:]=0
    for jj in range(numdat):
        for ii in range(numpar):
            ttrand[jj,ii]=ttrange[ii,0]+random.random()*(ttrange[ii,1]-ttrange[ii,0])
# here we use scipy to find the closest local minimum to the random data point generated just above
# if problems occur using a custom ExFun(), conisder changing the method of minimum search below to e.g. Nelder-Mead
        cgmin=spmin(ExFun,ttrand[jj,:],method='L-BFGS-B',options={'gtol': 1e-7, 'eps':1e-7})
#        cgmin=spmin(ExFun,ttrand[jj,:],method='Nelder-Mead')
#        cgmin=spmin(ExFun,ttrand[jj,:],method='CG')
        if cgmin.success:
            locmin[jj,:]=cgmin.x
            locminval[jj]=cgmin.fun
            oor=0 # checks whether the found local minimum is out of range
            for ii in range(numpar):
                if locmin[jj,ii]<ttrange[ii,0] or locmin[jj,ii]>ttrange[ii,1]:
                    oor=1
            if oor==1:
# here we tag out of range minima as LDup=2; note that these are stored but
# not tested for duplicates in this version
                LDup[jj]=2
        else:
# here we tag failed random searches as LDup=-1
            LDup[jj]=-1
# here we identify duplicate minima and tag them as LDup=1
    for ii in range(0,numdat-1):
        for jj in range(ii+1,numdat):
            if (LDup[ii]==0 and LDup[jj]==0 and np.linalg.norm(locmin[jj,:]-locmin[ii,:])<locmintol):
                LDup[jj]=1
                DupPos[jj]=ii+1 # note the offset of +1 which is needed for combining FullRand() with EffSphereSearch()
# here we count the independent local minima, which are tagged as LDup=0
    LMCount=0
    for ii in range(numdat):
        if LDup[ii]==0:
            LMCount+=1
# here we gather the data in the outputfile as the custom object class LocMinArr
    output=LocMinArr(locmin, locminval, ttrand, LDup, DupPos, LMCount)
    return output

def EffSphereSearch(numdat,numpar,ttrange,LocMinRand):
    '''
    EffSphereSearch() continues the minimum search using an active learning
    procedure as follows. We analyse the result of the random search stored
    in LocMinRand by calculating the distribution of the straight line distance
    between the random starting points and the corresponding nearest local
    minima, then compute the mean value R_M. We use this as the approximate
    mean effective attraction radius of the local minima. We then compute
    the volume volSphere of the numpar dimensional sphere of radius R_M and
    compare it to the volume volParSpace of the sampled parameter space in
    order to make a decision on how to proceed with the minimum search. If
    volSphere/volParSpace<0.1 we expect that the local minima are uniformly
    scattered about in the parameter space and proceed with a random search with
    the constraint that the new queries should be made sufficiently far from the
    previosuly found local minima. In the opposite case we assume that the minima
    are more likely to be clustered, and we proceed with a random search with
    the constraint that the new queries should be made sufficiently close to
    the previosuly found local minima.
    '''
    R_M=0.0
    numaver=0
    for ii in range(numdat):
        if LocMinRand.LDup[ii]==0 or LocMinRand.LDup[ii]==1:
            R_M+=np.linalg.norm(LocMinRand.LocMin[ii]-LocMinRand.MinOrig[ii])
            numaver+=1
    R_M=R_M/numaver
    effrad=R_M # the average effective attraction radius around local minima
    volParSpace=np.prod(ttrange[:,1]-ttrange[:,0]) # volume of parameter space
#    d_eff=2.0*((GammaFun((numpar/2.0)+1.0)*volParSpace/(onepi**(numpar/2.0)))**(1.0/numpar)) # effective diameter of the parameter space
    volSphere=(onepi**(numpar/2.0))*(effrad**numpar)/GammaFun((numpar/2.0)+1.0) # volume of one effective sphere in numpar dimensional space
    volSpheres=LocMinRand.LMCount*volSphere # sum of effective sphere volumes in numpar dimensional space
    print('Mean effective attraction volume around previously found local minima:',volSphere)
    print('This is',100.0*volSphere/volParSpace,'percent of the total parameter space, which itself has the volume:',volParSpace)
    if volSphere/volParSpace>0.1:
        intraES=1 # sets the search below to take place inside the scaled effective spheres
        print('Function classified as having clustered minima. Effective sphere search will commence near the spheres.')
    else:
        intraES=-1 # sets the search below to take place outside the scaled effective spheres
        print('Function classified as having uniformly scattered minima. Effective sphere search will commence away from the spheres.')
# here we scale the effective radius such that new points are queried in the inner 
# 50% or the outer 50% of the parameter space around the set of previously found local minima
# determined by the formula:       (scal**numpar)*volSpheres=0.5*volParSpace
    scal=(0.5*volParSpace/volSpheres)**(1.0/numpar)
    ttrand=np.zeros((numdat,numpar)) # random set of tt parameters for ExFun(tt)
    locmin=np.zeros((numdat,numpar)) # position of local minima in parameter space
    locminval=np.zeros((numdat)) # ExFun() values of local minima
    LDup=np.arange(0,numdat) # variable tracking whether the local minimum is unique (0), a duplicate (1), or if minimization failed (-1) or led outside the required range (2)
    DupPos=np.arange(0,numdat) # index of unique local minimum if the current local minimum is a duplicate; an offset of 1 is added for convenience
    LDup[:]=0
    DupPos[:]=0
    for jj in range(numdat):
        keep_looking=bool(True) # logical variable tracking whether to keep looking for suitable random points
        while keep_looking:
            for ii in range(numpar):
                ttrand[jj,ii]=ttrange[ii,0]+random.random()*(ttrange[ii,1]-ttrange[ii,0])
            npassed=0 # integer tracking how many local minima are far/close enough from/to the new random point
            for kk in range(numdat):
                if np.linalg.norm(ttrand[jj,:]-LocMinRand.LocMin[jj])*intraES<scal*effrad*intraES:
                    npassed+=1
            if npassed==numdat:
                keep_looking=bool(False)
# now we query the newly found set of random points
        cgmin=spmin(ExFun,ttrand[jj,:],method='L-BFGS-B',options={'gtol': 1e-7, 'eps':1e-7})
        if cgmin.success:
#            locmin[jj,:]=(np.concatenate((cgmin.x,cgmin.fun),axis=None))
            locmin[jj,:]=cgmin.x
            locminval[jj]=cgmin.fun
            oor=0 # checks whether the found local minimum is out of range
            for ii in range(numpar):
                if locmin[jj,ii]<ttrange[ii,0] or locmin[jj,ii]>ttrange[ii,1]:
                    oor=1
            if oor==1:
# here we tag out of range minima as LDup=2
                LDup[jj]=2
        else:
# here we tag failed random searches as LDup=-1
            LDup[jj]=-1
# here we detect duplicates between the new minima and the old minima
    for ii in range(0,numdat):
        for jj in range(0,numdat):
            if (LDup[jj]==0 and LocMinRand.LDup[ii]==0 and np.linalg.norm(LocMinRand.LocMin[ii,:]-locmin[jj,:])<locmintol):
                LDup[jj]=1
                DupPos[jj]=-ii-1 # this is why the offset is needed for DupPos in FullRand(), as otherwise the first index would be 0 and that cannot be distinguished from -0
# now we detect duplicate minima among the newly found local minima:
    for ii in range(0,numdat-1):
        for jj in range(ii+1,numdat):
            if (LDup[ii]==0 and LDup[jj]==0 and np.linalg.norm(locmin[jj,:]-locmin[ii,:])<locmintol):
                LDup[jj]=1
                DupPos[jj]=ii+1
# next we count all the new local minima:
    LMCount=0
    for ii in range(numdat):
        if LDup[ii]==0:
            LMCount+=1
# now we export the new local minima; note that when DupPos is negative, it
# refers to a previously found local minimum in LocMinRand
    output=LocMinArr(locmin, locminval, ttrand, LDup, DupPos, LMCount)
    return output

    
def main():
# select the following two lines to use the code with the 2-variable polynomial function in exfun.py 
#    nbudget=30 # the total budget
#    numpar=2
# select the following two lines to use the code with the 10-variable cosine product function in exfun.py 
    nbudget=1024 # the total budget
    numpar=10
    print('This is eMiLy.py 1.0.3 (January 2019).\nWritten by Viktor Zolyomi. Licenced as free software under the GNU GPL.')
    numdat=nbudget//2
    ttrange=np.zeros((numpar,2))
    ttrange[:,0]=-1.0
    ttrange[:,1]=1.0
    print('\nMinimum search of provided function with',numpar,'parameters will commence using a budget of',nbudget,'minimum queries')
    print('\nStage I. Initiating random search using',numdat,'queries')
    ttlocminRAND=FullRand(numdat,numpar,ttrange)
    print('Search complete. Found',ttlocminRAND.LMCount,'local minima')
    print('\nStage II. Initiating active learning search relying on effective attraction spheres using',numdat,'queries')
    ttlocminES=EffSphereSearch(numdat,numpar,ttrange,ttlocminRAND)
    print('Search complete. Found',ttlocminES.LMCount,'local minima')
    print('\nMinimum search complete. Found a total of',ttlocminRAND.LMCount+ttlocminES.LMCount,'local minima')
    with open('LocMin.dat','wt') as outputfile:
        outputfile.write('This is eMiLy.py 1.0.3 (January 2019)\nWritten by Viktor Zolyomi. Licenced as free software under the GNU GPL.\n')
        outputfile.write('\nMinimum search completed using a budget of %d minimum queries\n' % nbudget)
        outputfile.write('\nResults of Stage I. (random search):\n')
        outputfile.write('%d minima found.\n' % ttlocminRAND.LMCount)
        lcount=0
        for ii in range(numdat):
            if ttlocminRAND.LDup[ii]==0:
                outputfile.write('\n\nMinimum %d:\n' % lcount)
                outputfile.write('\nValue of function: %f\n' % ttlocminRAND.LocMinVal[ii])
                outputfile.write('\nList of parameters:\n')
                json.dump(ttlocminRAND.LocMin[ii,:].tolist(), outputfile, separators=(',', ':'), sort_keys=True, indent=4)
                lcount+=1
        outputfile.write('\n\nResults of Stage II. (active learning search relying on effective attraction spheres):\n')
        outputfile.write('%d minima found.\n' % ttlocminES.LMCount)
        lcount=0
        for ii in range(numdat):
            if ttlocminES.LDup[ii]==0:
                outputfile.write('\n\nMinimum %d:\n' % lcount)
                outputfile.write('\nValue of function: %f\n' % ttlocminES.LocMinVal[ii])
                outputfile.write('\nList of parameters:\n')
                json.dump(ttlocminES.LocMin[ii,:].tolist(), outputfile, separators=(',', ':'), sort_keys=True, indent=4)
                lcount+=1
    print('\nMinima saved in LocMin.dat')

if __name__ == "__main__":
    main()
