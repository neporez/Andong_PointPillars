import numpy as np
import torch
import sys

import tensorrt as trt
import pycuda.driver as cuda

import os

home_path = os.path.expanduser('~')

model_path = home_path+"/PointPillars"

sys.path.append(model_path)

from utils import keep_bbox_from_lidar_range
from model import PointPillarsPre, PointPillarsPos


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class PointPillarsTensorRT():

    def __init__(self,
                 nclasses=2,
                 voxel_size=[0.32,0.32,8],
                 point_cloud_range=[-69.12, -69.12, -3.0, 69.12, 69.12, 5.0],
                 max_num_points=16,
                 max_num_pillars=40000,
                 engine_path=None) :
        
        

        self.__nclasses = nclasses
        self.__voxel_size = voxel_size
        self.__point_cloud_range = point_cloud_range
        self.__max_num_points = max_num_points
        self.__max_num_pillars = max_num_pillars
        self.__engine_path = engine_path
        

        if self.__engine_path is None :
            print("trt engine is unavailable")
            return


        self.__model_pre = PointPillarsPre(voxel_size=self.__voxel_size, point_cloud_range=self.__point_cloud_range,max_num_points=self.__max_num_points).cuda()

        self.__trt_logger = trt.Logger(trt.Logger.WARNING)
        self.__trt_runtime = trt.Runtime(self.__trt_logger)

        self.__engine = self.__load_engine(self.__trt_runtime, self.__engine_path)

        self.__context_trt = self.__engine.create_execution_context()

        self.__inputs, self.__outputs, self.__bindings, self.__stream = self.__allocate_buffers(self.__max_num_pillars, self.__max_num_points)
        
        self.__model_post = PointPillarsPos(nclasses=self.__nclasses).cuda()

        self.__model_pre.eval()
        self.__model_post.eval()



    def __load_engine(self, trt_runtime, engine_path) :
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    

    def __allocate_buffers(self, max_num_pillars, max_num_points):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in range(self.__engine.num_io_tensors):
            tensor_name = self.__engine.get_tensor_name(binding)
            shape = self.__engine.get_tensor_shape(tensor_name)

            if -1 in shape:
                if 'input_pillars' in tensor_name:
                    shape = (max_num_pillars, max_num_points, 4)
                elif 'input_coors_batch' in tensor_name:
                    shape = (max_num_pillars, 4)
                elif 'input_npoints_per_pillar' in tensor_name:
                    shape = (max_num_pillars,)
                else:
                    shape = self.__engine.get_tensor_profile_shape(tensor_name, 0)[2]

            size = trt.volume(shape)
            dtype = trt.nptype(self.__engine.get_tensor_dtype(tensor_name))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.__engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream


    def __do_inference(self, context, bindings, inputs, outputs, stream):
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        stream.synchronize()
        return [out.host for out in outputs]
    

    
    
    def inference(self, points) :

        
        points[:, 3] /= 255.0
    
        pc_torch = torch.from_numpy(points).cuda()
        

        with torch.no_grad():
            pillars, coors_batch, npoints_per_pillar = self.__model_pre(batched_pts=[pc_torch])

            num_pillars = pillars.shape[0]

            if(num_pillars < 50):
                print("num_pillars < 50 , num_pillars: ", num_pillars)
                return None, None, None 

        for i, inp in enumerate(self.__inputs):
            if i == 0:
                data = pillars.cpu().numpy().astype(np.float32)
            elif i == 1:
                data = coors_batch.cpu().numpy().astype(np.int32)
            elif i == 2:
                data = npoints_per_pillar.cpu().numpy().astype(np.int32)
            else:
                raise ValueError(f"Unexpected input index: {i}")
            if data.dtype != inp.host.dtype:
                print(f"Warning: Data type mismatch for input {i}. Expected {inp.host.dtype}, got {data.dtype}")
                data = data.astype(inp.host.dtype)
            np.copyto(inp.host[:data.size], data.ravel())
        self.__context_trt.set_binding_shape(0, (num_pillars, self.__max_num_points, 4))
        self.__context_trt.set_binding_shape(1, (num_pillars, 4))
        self.__context_trt.set_binding_shape(2, (num_pillars,))
        trt_outputs = self.__do_inference(self.__context_trt, self.__bindings, self.__inputs, self.__outputs, self.__stream)
        result = [torch.from_numpy(trt_outputs[0].reshape(-1, 8+self.__nclasses)).cuda()]
        try:
            result_filter = self.__model_post(result)[0]
        except:
            print("inference failed")
            return None, None, None
        result_filter = keep_bbox_from_lidar_range(result_filter, np.array(self.__point_cloud_range, dtype=np.float32))
        lidar_bboxes = result_filter['lidar_bboxes']
        labels, scores = result_filter['labels'], result_filter['scores']

        return lidar_bboxes, labels, scores
        
        
