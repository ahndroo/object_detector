"""
    Convert the PeopleNet ONNX file to TensorRT engine
"""

import os
import sys

import tensorrt as trt

def main():

    onnx_filepath = "/detector/models/peoplenet/resnet34_peoplenet.onnx"
    trt_filepath = "/detector/models/peoplenet/model.trt"

    cmd = "/usr/src/tensorrt/bin/trtexec --onnx=/detector/models/peoplenet/resnet34_peoplenet.onnx --saveEngine=/detector/models/peoplenet/model.trt"
    os.system(cmd)

    # idk how to do this with Python --figure later
    # # Initialize the TensorRT engine
    # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    # builder = trt.Builder(TRT_LOGGER)
    # network = builder.create_network()
    # parser = trt.OnnxParser(network, TRT_LOGGER)

    # # Parse the ONNX file
    # with open(onnx_filepath, 'rb') as model:
    #     if not parser.parse(model.read()):
    #         for error in range(parser.num_errors):
    #             print(parser.get_error(error))
    #         return

    # # Build and serialize the TensorRT engine
    # builder.max_workspace_size = 1 << 30
    # builder.max_batch_size = 1
    # engine = builder.build_cuda_engine(network)
    # with open(trt_filepath, 'wb') as f:
    #     f.write(engine.serialize())

    # print("TensorRT engine saved at:", trt_filepath)

if __name__ == "__main__":
    main()