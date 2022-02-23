#!/usr/bin/env python3.6.8
# -*- coding: utf-8 -*-
# Copyright:    Yuhan Jiang, Ph.D.(http://www.yuhanjiang.com)
# Date:         2/22/2022
# Discriptions : pointcloud to orthoimage for building facade
# Major updata : 
import copy
import gc
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
from mpl_toolkits import mplot3d
import cv2 as cv
from scipy.interpolate import griddata
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from itertools import repeat
from multiprocessing import Process
from multiprocessing import pool


class NoDaemonProcess(Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)
# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(pool.Pool):
    Process = NoDaemonProcess
def rotate(img,angle):
    h, w =img.shape[0:2]
    center = (w/2, h/2)
    M = cv.getRotationMatrix2D(center, angle, 1)
    rotated = cv.warpAffine(img,M,(w,h))
    return rotated
def newdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        #print(path+'   Successful')
        return True
    else:
        #print(path+'   Exists')
        return False
def generateGridImageUisngMultiCPU(X,Y,Z):
    x_range=((X.max()-X.min()))
    y_range=((Y.max()-Y.min()))

    grid_x,grid_y=np.mgrid[X.min():X.max():(x_range*1j),Y.min():Y.max():(y_range*1j)]  # create the grid size

    grid_z = griddata((X,Y), Z, (grid_x, grid_y), method='linear')#{‘linear’, ‘nearest’, ‘cubic’}, optional
    try:
        return grid_z
    finally:
        del X,Y,Z,grid_z,grid_x,grid_y
        gc.collect()
def PointCloud2Orthoimage2(points,colors,downsample=10,GSDmm2px=5):
    print('[PointCloudShape XYZ RGB]',points.shape,colors.shape)
    if downsample>0:
        X=points[:,1][::downsample]*-1000/GSDmm2px  # 1000 means:1mm to 1 px
        Y=points[:,0][::downsample]*1000/GSDmm2px  # [::10] downsample 1/10
        Z=points[:,2][::downsample]*1000# elevation in mm
        R=(colors[:,0][::downsample])# keep 16-bit
        G=(colors[:,1][::downsample])
        B=(colors[:,2][::downsample])
        print('[DownSamplePCShape]',X.shape,Y.shape,Z.shape)
    else:
        X=points[:,1]*-1000/GSDmm2px  # 1000 means:1mm to 1 px
        Y=points[:,0]*1000/GSDmm2px  # [::10] downsample 1/10
        Z=points[:,2]*1000#elevation in mm
        R=colors[:,0]# keep 16-bit
        G=colors[:,1]
        B=colors[:,2]
    print("[RGBColorRange]",R,G,B)
    x_range=((X.max()-X.min()))
    y_range=((Y.max()-Y.min()))
    print("[ImageFrameSize]",x_range,y_range)
    ele_max=Z.max()
    ele_min=Z.min()
    z_range=[ele_min,ele_max]
    print('[x,y,z range in mm]',x_range*GSDmm2px,y_range*GSDmm2px,z_range)

    EleRGB=[Z,R,G,B]
    pool=MyPool(4)
    grid_Mutiple=pool.starmap(generateGridImageUisngMultiCPU,zip(repeat(X),repeat(Y),EleRGB))
    pool.close()

    grid_ele=grid_Mutiple[0].astype('float')
    grid_R=grid_Mutiple[1].astype('uint16')
    grid_G=grid_Mutiple[2].astype('uint16')
    grid_B=grid_Mutiple[3].astype('uint16')

    grid_RGB=np.zeros((grid_ele.shape[0],grid_ele.shape[1],3)).astype('uint16')
    grid_RGB[:,:,0]=grid_R
    grid_RGB[:,:,1]=grid_G
    grid_RGB[:,:,2]=grid_B

    print('[RGB,Ele imageShape]',grid_RGB.shape,grid_ele.shape)
    print('[GSD: mm/px]',GSDmm2px)
    try:
        return grid_RGB,grid_ele,[ele_min,ele_max]
    finally:
        del grid_B,grid_G,grid_R,grid_RGB,grid_ele,grid_Mutiple,EleRGB,X,Y,Z,R,G,B,pool,points,colors


def cameraSelector(v):
    camera=[]
    camera.append(v.get('eye'))
    camera.append(v.get('phi'))
    camera.append(v.get('theta'))
    camera.append(v.get('r'))
    return np.concatenate(camera).tolist()

def vector_angle(u, v):
    return np.arccos(np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v)))

def main(glb_file_path,pointName='5mm_18_34_56',downsample=10,GSDmm2px=5,bool_alignOnly=False,b='win',bool_generate=False):
    print('$',pointName)
    bool_confirm=False
    if b=='win':
        #import pptk
        import open3d as o3d
        axis_mesh=o3d.geometry.TriangleMesh.create_coordinate_frame()  #o3d.geometry.TriangleMesh.create_mesh_coordinate_frame(size=5.0,origin=np.array([0.,0.,0.]))
        pcd=o3d.io.read_point_cloud(glb_file_path+pointName+'.pts', format='xyzrgb') # xyz(double)rgb(256)normal(double)
        points=(np.asarray(pcd.points))
        colors=(np.asarray(pcd.colors))
        #region get_the_min_rotated_boundingbox
        pcd_t=o3d.geometry.PointCloud()
        pcd_t.points=o3d.utility.Vector3dVector(points)
        pcd_t.colors=o3d.utility.Vector3dVector(colors/256)
        #pcd.normals=o3d.utility.Vector3dVector(normals)
        o3d.visualization.draw_geometries([pcd_t,axis_mesh],window_name='OrginalPCD_WC'+pointName,width=1920//3*2,height=1080//3*2)
        if bool_alignOnly and bool_generate:
            print('[Start]---/...')
            #o3d.io.write_point_cloud(glb_file_path+pointName+"aligned.pcd",pcd_t)
            df=np.hstack([np.array(pcd_t.points),np.asarray(pcd_t.colors)])
            df=pd.DataFrame(df)
            df.to_csv(glb_file_path+pointName+"aligned.csv",index=False,header=False)
            print('[Saved]',glb_file_path+pointName+"aligned.csv")
    pc=pd.read_csv(glb_file_path+pointName+"aligned.csv",index_col=False,header=None)
    pc=np.array(pc)
    print(['CSV pointcloud formate'],pc.shape)
    points=pc[:,0:3]#*-1
    colors=pc[:,3:6]
    #endregion
    if bool_alignOnly:
        print('[Piontcloud aligment only]')
        return False
    #region ouput
    grid_RGB,grid_ele,(ele_min,ele_max)=PointCloud2Orthoimage2(np.array(points),np.asarray(colors)*65535,downsample=downsample,GSDmm2px=GSDmm2px)
    grid_RGB=(grid_RGB/(2**16-1)*255).astype('uint8')
    grid_map=((grid_ele-ele_min)/(ele_max-ele_min)*255).astype('uint8')
    #if grid_ele.shape[0]>grid_ele.shape[1]:# alway keep width larger than height
    #    grid_ele=rotate(grid_ele,90)
    #    grid_RGB=rotate(grid_RGB,90)
    try:
        return grid_RGB,grid_ele,grid_map,(ele_min,ele_max),GSDmm2px
    finally:
        newdir(glb_file_path+'/Demo/'+pointName+'/')
        cv.imwrite(glb_file_path+'/Demo/'+pointName+'/'+pointName+'RGB.jpg',cv.cvtColor(grid_RGB,cv.COLOR_RGB2BGR),[int(cv.IMWRITE_JPEG_QUALITY),100])
        cv.imwrite(glb_file_path+'/Demo/'+pointName+'/'+pointName+'DEM.jpg',grid_map,[int(cv.IMWRITE_JPEG_QUALITY),100])
        print('[Done]',glb_file_path+'/Demo/'+pointName+'/'+pointName+'RGB/DEM.jpg')
        try:
            del points,colors,pcd
        except:
            print("Skip del temp files")
        gc.collect()

#-------
if __name__ == '__main__':
    if os.path.exists('D:/'):
        glb_file_path='D:/CentOS/Model/'  # screenshot saving path
        b='win'
        cpu=3
        import open3d as o3d
    elif os.path.exists('/data/'):
        glb_file_path='/data/Model/'  # screenshot saving path
        b='server'
        cpu=4
    PC_Name=["S_right"]
    #PC_Name.reverse()
    if b=='win':
        for i in PC_Name:
            main(pointName=i,glb_file_path=glb_file_path,GSDmm2px=5,bool_alignOnly=1,b=b,bool_generate=1,downsample=-1)# default 5
    else:
        for i in PC_Name:
            main(pointName=i,glb_file_path=glb_file_path,GSDmm2px=23,bool_alignOnly=False,b=b,downsample=-1)

