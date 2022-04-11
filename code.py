# -*- coding: utf-8 -*-
"""
Created on Fri Apr 8 21:03:12 2022

@author: Agamdeep S. Chopra
Acknowledgement: I utilized lecture notes, recommended texts, and various online resources as reference to write this code.
                 Voxel coloring was hardcoded using lookup table. If time was not a constraint, surface normal based coloring would have been explored.
      
"""
import numpy as np
import cv2
import polyscope as ps
from numba import jit


@jit(nopython=False)
def save_surface_point_cloud(points, color, path, file_name): # (N,3),(N,3),str   
    fout = open(path+'\\'+file_name+'.ply', 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex %d\n"%(points.shape[0]))
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property uchar red\n")
    fout.write("property uchar green\n")
    fout.write("property uchar blue\n")
    fout.write("element face 0\n")
    fout.write("end_header\n")
    for i in range(points.shape[0]):
        fout.write('%f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],color[i,0],color[i,1],color[i,2]))
    fout.close() 


@jit(nopython=False)
def best_idx(imgs,encode):
    lkt = [[1,1,1,1],[1,0,1,2],[0,1,1,7],[0,0,1,6],[1,1,0,0],[1,0,0,3],[0,1,0,0],[0,0,0,4]] # Simplified lookup table
# =============================================================================
# REMARKS:
#     Lookup table not efficient way. 
#     Only this combination seems to give a decent estimation. 
#     Adding z coord to estimation might improve color estimation.
#     Although, this is no way an ideal way to solve this.
#     If I had more time, I would calculate the surface normals and use that info,-
#     -paired with the camera center coordinates to calculate- 
#     -which surface voxels should be colored using which camera.
# =============================================================================    
    for i in range(len(lkt)):
        if encode[0] == lkt[i][0] and encode[1] == lkt[i][1] and encode[2] == lkt[i][2]:
            best_idx = lkt[i][3]   
    return best_idx


@jit(nopython=False)
def hw3(voxels):    
    path = 'R:\classes 2020-22\Spring 2022\CS 532 3D CV\HW3\cs532_HW03'
    P = np.zeros((3,4,8))
    S = np.zeros((582, 780, 8)) #np.zeros((582, 780, 3, 8))
    I = np.zeros((582, 780, 3, 8))
    rawP = np.asarray([[776.649963,-298.408539,-32.048386,993.1581875,132.852554,120.885834,-759.210876,1982.174000,0.744869,0.662592,-0.078377,4.629312012],
                       [431.503540,586.251892,-137.094040,1982.053375,23.799522,1.964373,-657.832764,1725.253500,-0.321776,0.869462,-0.374826,5.538025391],
                       [-153.607925,722.067139,-127.204468,2182.4950,141.564346,74.195686,-637.070984,1551.185125,-0.769772,0.354474,-0.530847,4.737782227],
                       [-823.909119,55.557896,-82.577644,2498.20825,-31.429972,42.725830,-777.534546,2083.363250,-0.484634,-0.807611,-0.335998,4.934550781],
                       [-715.434998,-351.073730,-147.460815,1978.534875,29.429260,-2.156084,-779.121704,2028.892750,0.030776,-0.941587,-0.335361,4.141203125],
                       [-417.221649,-700.318726,-27.361042,1599.565000,111.925537,-169.101776,-752.020142,1982.983750,0.542421,-0.837170,-0.070180,3.929336426],
                       [94.934860,-668.213623,-331.895508,769.8633125,-549.403137,-58.174614,-342.555359,1286.971000,0.196630,-0.136065,-0.970991,3.574729736],
                       [452.159027,-658.943909,-279.703522,883.495000,-262.442566,1.231108,-751.532349,1884.149625,0.776201,0.215114,-0.592653,4.235517090]])
    for i in range(8):
        for j in range(3):
            P[j,:,i] = rawP[i, 4*j : 4*j + 4]                  
    for i in range(8):    
        S[:,:,i] = np.where(cv2.imread(path+'\silh_cam0%d_00023_0000008550.pbm'%(i))[:,:,0] > 0, 1, 0) # <- convert silh. to binary
        I[:,:,:,i] = cv2.imread(path+'\cam0%d_00023_0000008550.png'%(i))    
    vox_grid = np.asanyarray([[-2.5,2.5],[-3.,3.],[0.,2.5]])
    grid_mid = [0.3,-1.4,1.0] # midpoints manually extrapolated. not ideal way to solve this problem.
    gridX,gridY,gridZ = vox_grid[0,1] - vox_grid[0,0], vox_grid[1,1] - vox_grid[1,0], vox_grid[2,1] - vox_grid[2,0]
    tot_vol = gridX * gridY * gridZ
    vox_len = (tot_vol / voxels)**(1/3)
    pos = []
    col = []
    flag = True
    pt_hist = None
    e = [0,0,0]   
    for z in np.arange(vox_grid[2,0],vox_grid[2,1],vox_len):
        e[2] = 1 if z <= grid_mid[2] else 0        
        for x in np.arange(vox_grid[0,0],vox_grid[0,1],vox_len):          
            e[0] = 1 if x >= grid_mid[0] else 0        
            for y in np.arange(vox_grid[1,0],vox_grid[1,1],vox_len):                
                e[1] = 1 if y <= grid_mid[1] else 0    
                S_ppx_pc = np.zeros(8) #<-- 0 when silh. of image[i] = 0 and 1 when silh. is 1...
                H_c = np.asarray((x,y,z,1.)) # <-- (4,)              
                for i in range(8):                   
                    X = P[:,:,i] @ H_c  # (3,4) @ (4,) -> (3,) ... transf. coords. ...
                    X = np.round(X / X[-1]) # (x,y,1) ...img plane...         
                    if (X[0] >= 0 and X[1] >= 0) and (X[0] < I.shape[1] and X[1] < I.shape[0]): 
                        S_ppx_pc[i] = S[round(X[1]),round(X[0]),i]                        
                if np.all(S_ppx_pc == 1): #<- save the first nonempty voxel...
                    if flag:
                        pos.append([x,y,z])
                        col.append(I[round(X[1]),round(X[0]), :, best_idx(I,e)][::-1])
                        flag = False
                    else: #<- keep previous nonempty voxel following the first nonempty voxel in memory...
                        pt_hist = [[x,y,z],I[round(X[1]),round(X[0]), :, best_idx(I,e)][::-1]]
                elif not(flag):#<- save last nonempty voxel and reset parameters...
                    if pt_hist != None:
                        pos.append(pt_hist[0])
                        col.append(pt_hist[1])
                    flag = True
                    pt_hist = None                                                   
    pos, col = np.asanyarray(pos), np.asanyarray(col)
    path = 'R:\classes 2020-22\Spring 2022\CS 532 3D CV\HW3'    
    save_surface_point_cloud(pos, col, path, "colored_surface_points_temp")  
    ps.init()
    ps_cloud = ps.register_point_cloud("my points", pos, enabled=True)
    ps_cloud.add_color_quantity("my colors", col/255)
    ps.show()
   
    
def main():
    vox = input('Enter integer # of voxels:  ')
    vox = float(vox.replace(',',''))
    hw3(vox)
    

if __name__ == '__main__':
    main()