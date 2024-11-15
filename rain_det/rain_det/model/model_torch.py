import numpy as np
import torch
import sys

import os

home_path = os.path.expanduser('~')

model_path = home_path+"/PointPillars"

sys.path.append(model_path)

from utils import keep_bbox_from_lidar_range
from model import PointPillarsPre, PointPillarsPos, PointPillarsCore



class PointPillarsTorch():

    def __init__(self,
                 nclasses=2,
                 voxel_size=[0.32,0.32,8],
                 point_cloud_range=[-69.12, -69.12, -3.0, 69.12, 69.12, 5.0],
                 max_num_points=16,
                 max_num_pillars=40000,
                 torch_path=None) :
        
        

        self.__nclasses = nclasses
        self.__voxel_size = voxel_size
        self.__point_cloud_range = point_cloud_range
        self.__max_num_points = max_num_points
        self.__max_num_pillars = max_num_pillars
        self.__torch_path = torch_path
        

        if self.__torch_path is None :
            print("torch engine is unavailable")
            return


        self.__model_pre = PointPillarsPre(voxel_size=self.__voxel_size, point_cloud_range=self.__point_cloud_range,max_num_points=self.__max_num_points).cuda()
        
        self.__model_core = PointPillarsCore(nclasses=self.__nclasses, voxel_size=self.__voxel_size,point_cloud_range=self.__point_cloud_range).cuda()

        self.__model_core.load_state_dict(torch.load(self.__torch_path))

        self.__model_post = PointPillarsPos(nclasses=self.__nclasses).cuda()

        self.__model_pre.eval()
        self.__model_core.eval()
        self.__model_post.eval()

   
    def inference(self, points) :

        points[:, 3] /= 255.0
        
        pc_torch = torch.from_numpy(points).cuda()
        

        with torch.no_grad():
            pillars, coors_batch, npoints_per_pillar = self.__model_pre(batched_pts=[pc_torch])

            num_pillars = pillars.shape[0]

            if(num_pillars < 50):
                print("num_pillars < 50 , num_pillars: ", num_pillars)
                return None, None, None

        
        result = self.__model_core(pillars, coors_batch, npoints_per_pillar, mode='val')
        try:
            result_filter = self.__model_post(result)[0]
        except:
            print("inference failed")
            return None, None, None
        result_filter = keep_bbox_from_lidar_range(result_filter, np.array(self.__point_cloud_range, dtype=np.float32))
        lidar_bboxes = result_filter['lidar_bboxes']
        labels, scores = result_filter['labels'], result_filter['scores']
        
        return lidar_bboxes, labels, scores
        

        