import sys
import cv2
import random
import matplotlib
import numpy as np
#matplotlib.use('Agg')
from tqdm import tqdm
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData, PlyElement

# path through the algorithm to import "spherical_algo"
path='C:/Users/buntinx/Dropbox/scanvan/python/scanvan_algorithm'
sys.path.insert(0,path)
import spherical_algo

# read a file with 3d points
def importation_data(name):
    fichier=open(name+'.txt','r')
    text=fichier.read()
    fichier.close()
    text=text.split('\n')
    data=[]
    for i in range(len(text)):
        els=text[i].split('\t')
        data.append(np.array([float(els[0]),float(els[1]),float(els[2])]))
    return data

# project points on a spherical camera with ray equal to 1
def projection(p3d,center):
    res=[]
    for i in range(len(p3d)):
        vec=np.subtract(p3d[i],center)
        vec/=np.linalg.norm(vec)
        point=vec
        res.append(point)
    return res

# generate three spherical camera projections from one 3d set points
def generate_unit_spheres(p3d,centers):
    p3d_1=projection(p3d,centers[0])
    p3d_2=projection(p3d,centers[1])
    p3d_3=projection(p3d,centers[2])
    return p3d_1,p3d_2,p3d_3

# save a 3d scene into a ply file
def save_ply(scene,name):
    scene_ply=[]
    for elem in scene:
        scene_ply.append(tuple(elem))
    scene_ply=np.array(scene_ply,dtype=[('x','f4'),('y','f4'),('z','f4')])
    el=PlyElement.describe(scene_ply,'vertex',comments=[name])
    PlyData([el],text=True).write(name+'.ply')

# verify that the rotation is without relexions and provide a good solution
def svd_rotation(v,u):
    vu=np.dot(v,u)
    det=round(np.linalg.det(vu),4)
    m=np.identity(3)
    m[2,2]=det
    vm=np.dot(v,m)
    vmu=np.dot(vm,u)
    return vmu

# transform a set of point in order to match the scale and orientation of the second one (easier to compare)
def miseaechelle(data1,data2):
    if len(data1)==len(data2):
        longeur=len(data1)
    sv_corr_12=np.zeros((3,3))
    sv_cent_1=np.zeros(3)
    sv_cent_2=np.zeros(3)
    for i in range(longeur):
        sv_cent_1+=data1[i]
        sv_cent_2+=data2[i]
    sv_cent_1/=longeur
    sv_cent_2/=longeur
    scale_1=0.0
    scale_2=0.0
    for i in range(longeur):
        sv_diff_1=data1[i]-sv_cent_1
        sv_diff_2=data2[i]-sv_cent_2
        scale_1+=np.linalg.norm(sv_diff_1)
        scale_2+=np.linalg.norm(sv_diff_2)
        sv_corr_12+=np.outer(sv_diff_1,sv_diff_2)
    svd_U_12,svd_s_12,svd_Vt_12=np.linalg.svd(sv_corr_12)
    rotation=svd_rotation(svd_Vt_12.transpose(),svd_U_12.transpose())
    translation=sv_cent_2-np.dot(rotation,sv_cent_1)
    scale_factor=scale_2/scale_1
    new_data=[]
    for i in range(longeur):
        point=translation+scale_factor*np.dot(rotation,data1[i])
        new_data.append(point)
    return new_data

# data importation and generation of "wise" triplet centers
# name_in is bunny_complete or bunny_partial (partial is quicker)
name_in='bunny_partial'
name_out='bunny_partial'
data=importation_data(name_in)
x_min=10
x_max=-10
y_min=10
y_max=-10
z_min=10
z_max=-10
for point in data:
    x_min=min(x_min,point[0])
    x_max=max(x_max,point[0])
    y_min=min(y_min,point[1])
    y_max=max(y_max,point[1])
    z_min=min(z_min,point[2])
    z_max=max(z_max,point[2])

# these centers can be set differently for testing other triplet center
c1=np.array([0.5*x_max,0.5*y_max,0.5*z_min])
c2=np.array([0.5*x_min,0.5*y_max,0.5*z_max])
c3=np.array([0.5*x_max,0.5*y_min,0.5*z_max])

# plot the initial situation with matplotlib
##fig=plt.figure()
##ax=fig.add_subplot(111,projection='3d')
##for i in range(len(data)):
##    ax.scatter(data[i][0],data[i][1],data[i][2],color='blue',s=0.5)
##ax.scatter(c1[0],c1[1],c1[2],color='red',s=4.0)
##ax.scatter(c2[0],c2[1],c2[2],color='green',s=4.0)
##ax.scatter(c3[0],c3[1],c3[2],color='purple',s=4.0)
##plt.show()

# generate the 3 spheres corresponding to data collected by the camera on three different centers
s1,s2,s3=generate_unit_spheres(data,[c1,c2,c3])
# call of the spherical algorithm
x=spherical_algo.pose_estimation(s1,s2,s3,10**-10,10000)
# output is transformed to be easily compared
new_data=miseaechelle(x[0],data)
# the original 3d points is generated in ply
save_ply(data,name_out+'_ori')
# the estimated 3d points is generated in ply
save_ply(new_data,name_out+'_est')
# a given measure of the "real" situation
result_ori=np.linalg.norm(c3-c2)/np.linalg.norm(c2-c1)
# the same measure of the estimated situation
result_est=np.linalg.norm(x[1][2]-x[1][1])/np.linalg.norm(x[1][1]-x[1][0])
# the two measure should be the same in the ideal case
print(result_ori)
print(result_est)

# plot the estimated situation with matplotlib
##fig=plt.figure()
##ax=fig.add_subplot(111,projection='3d')
##for point in x[0]:
##    ax.scatter(point[0],point[1],point[2],color='red',s=0.5)
##plt.show()
