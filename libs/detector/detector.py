
"""
This module is responsible for detecting the object in the image using
NVIDIA TAO Toolkit.  The model is converted to TensorRT

Current model being used is:
    TBD

Detects the following classes:
- People
/usr/src/tensorrt/bin/trtexec --onnx=/detector/models/peoplenet/resnet34_peoplenet.onnx --saveEngine=/detector/models/peoplenet/model.trt
"""

import os
import cv2
from .utils import *

import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from cuda import cudart

import torch
from torchvision.ops import nms

import pdb

TRT_LOGGER = trt.Logger()

class Detector:
    """_summary_
    """
    def __init__(self, model_file):
        self.model_file = model_file

        # Load the TensorRT model
        self.load_engine(model_file)

        # Set bindings
        self.setup_bindings()

        # Get input shape
        self.input_shape = self.inputs[0]["shape"] # NCHW

        # Class list - just hardcode this thang for now
        self.classes = ["person", "bag", "face"]


    def preprocess(self, frame):
        """
        Perform 8bit normalization and mean subtraction

        Args:
            frame (_type_): _description_
        """

        # Resize the frame
        frame = cv2.resize(frame, (self.input_shape[3], self.input_shape[2]))

        # Rescale the frame
        frame = frame / 255.0

        # Mean subtraction
        frame = frame - frame.mean()

        # Reshape frame
        frame = np.expand_dims(frame.transpose((2, 0, 1)), 0)

        return frame.astype(np.float32)

    def load_engine(self, engine_file_path):
        """


        Args:
            engine_file_path (_type_): _description_

        Returns:
            idkman: loaded trt model duh
        """
        assert os.path.exists(engine_file_path)
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context


    def setup_bindings(self):
        """
            Allocate memory for the input and output tensors
        """

        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
            shape = self.context.get_binding_shape(i)
            if is_input and shape[0] < 0:
                assert self.engine.num_optimization_profiles > 0
                profile_shape = self.engine.get_profile_shape(0, name)
                assert len(profile_shape) == 3  # min,opt,max
                # Set the *max* profile as binding shape
                self.context.set_binding_shape(i, profile_shape[2])
                shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda_call(cudart.cudaMalloc(size))
            host_allocation = None if is_input else np.zeros(shape, dtype)
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
            print("{} '{}' with shape {} and dtype {}".format(
                "Input" if is_input else "Output",
                binding['name'], binding['shape'], binding['dtype']))


    def infer(self, frame):
        """
        Execute inference on a batch of images.
        :param batch: A numpy array holding the image batch.
        :return A list of outputs as numpy arrays.
        """

        # Copy I/O and Execute
        memcpy_host_to_device(self.inputs[0]['allocation'], frame)
        self.context.execute_v2(self.allocations)
        for o in range(len(self.outputs)):
            memcpy_device_to_host(self.outputs[o]['host_allocation'], self.outputs[o]['allocation'])
        return [o['host_allocation'] for o in self.outputs]


    @staticmethod
    def divide_and_round_up(a, b):
        return (a + b - 1) // b
    
    @staticmethod
    def clip(value, min_val, max_val):
        return max(min(value, max_val), min_val)


    def postprocess(self, frame, outputs):
        """
        Postprocess the output of the model
        :param frame: The original frame
        :param outputs: The output of the model
        :return: The frame with the detections


        NOTE Assumes batch dimension == 1
        """

        bbox_layer_index = -1
        cov_layer_index = -1
        class_mismatch_warn = False
        object_list = []

        # Find bbox layer
        for i, layer in enumerate(self.outputs):
            if layer["name"] == "output_bbox/BiasAdd:0":
                bbox_layer_index = i
                bbox_layer_dims = layer["shape"]
                break
        if bbox_layer_index == -1:
            raise ValueError("Could not find bbox layer buffer while parsing")
        
        # Find cov layer
        for i, layer in enumerate(self.outputs):
            if layer["name"] == "output_cov/Sigmoid:0":
                cov_layer_index = i
                cov_layer_dims = layer["shape"]
                break
        if cov_layer_index == -1:
            raise ValueError("Could not find cov layer buffer while parsing")

        num_classes_to_parse = len(self.classes)
        grid_w, grid_h = cov_layer_dims[3], cov_layer_dims[2]
        bbox_norm_x, bbox_norm_y = 35.0, 35.0
        stride_x = self.divide_and_round_up(self.input_shape[3], bbox_layer_dims[3])
        stride_y = self.divide_and_round_up(self.input_shape[2], bbox_layer_dims[2])

        gc_centers_x = (np.arange(grid_w) * stride_x + 0.5) / bbox_norm_x
        gc_centers_y = (np.arange(grid_h) * stride_y + 0.5) / bbox_norm_y

        for cl in range(num_classes_to_parse):

            for y in range(grid_h):
                for x in range(grid_w):
                    cov = outputs[cov_layer_index][0][cl][y][x]
                    if cov < 0.4:
                        continue
                    bbox = outputs[bbox_layer_index][0][cl * 4:(cl + 1) * 4, y, x]
                    
                    x_min = (bbox[0] - gc_centers_x[x]) * -bbox_norm_x
                    y_min = (bbox[1] - gc_centers_y[y]) * -bbox_norm_y
                    x_max = (bbox[2] + gc_centers_x[x]) * bbox_norm_x
                    y_max = (bbox[3] + gc_centers_y[y]) * bbox_norm_y

                    object_list.append(
                        {
                            "class": cl,
                            "confidence": cov,
                            "bbox": [
                                self.clip(x_min, 0, self.input_shape[3] - 1),
                                self.clip(y_min, 0, self.input_shape[2] - 1),
                                self.clip(x_max, 0, self.input_shape[3] - 1) + 1,
                                self.clip(y_max, 0, self.input_shape[2] - 1) + 1
                            ]
                        }
                    )
        return object_list

    def detect(self, frame, nms_threshold=0.2):
        """
        Execute inference on frame
        :param frame: A numpy array of the frame.

        """

        # Preprocess the frame
        processed_frame = self.preprocess(frame)

        # Run inference
        outputs = self.infer(np.ascontiguousarray(processed_frame))
        detections = self.postprocess(frame, outputs)

        if len(detections) == 0:
            return frame

        # Get detection deets as array --should probably just not make a dictionary from this in postprocess
        labels = np.array([det["class"] for det in detections])
        bounding_boxes = np.array([det["bbox"] for det in detections])
        confs = np.array([det["confidence"] for det in detections])

        # Perform NMS
        keep_idx = nms(torch.tensor(bounding_boxes,dtype=torch.float32), torch.tensor(confs, dtype=torch.float32), nms_threshold)
        bounding_boxes = bounding_boxes[keep_idx]
        confs = confs[keep_idx]        
        labels = labels[keep_idx]

        if len(keep_idx) == 1:
            bounding_boxes = np.expand_dims(bounding_boxes, axis=0)
            confs = np.expand_dims(confs, axis=0)
            labels = np.expand_dims(labels, axis=0)

        # Bounding box scale factors
        input_w, input_h = self.input_shape[3], self.input_shape[2]  # (960, 544)
        frame_w, frame_h = frame.shape[1], frame.shape[0]  # (640, 480)
        w_sc = frame_w / input_w  # Scale width from model to frame
        h_sc = frame_h / input_h  # Scale height from model to frame
        scales = np.array([w_sc, h_sc, w_sc, h_sc])

        # Scale bounding boxes
        bounding_boxes = np.floor(bounding_boxes * scales).astype(np.int)

        # Draw bounding boxes
        for i, bbox in enumerate(bounding_boxes):
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, self.classes[labels[i]], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
