using System;
using System.IO;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace Micrograd.Core.Backends
{
    public class GpuBackend : ITensorBackend
    {
        private readonly CudaContext _context;
        private readonly CudaKernel _addKernel;
        private readonly CudaKernel _mulKernel;
        private readonly CudaKernel _tanhKernel;
        private readonly CudaKernel _reluKernel;
        private readonly CudaKernel _matmulKernel;
        private readonly CudaKernel _updateKernel;

        public GpuBackend()
        {
            _context = new CudaContext();
            var assemblyDir = Path.GetDirectoryName(typeof(GpuBackend).Assembly.Location)!;
            
            _addKernel = LoadKernel("TensorKernels.ptx", "TensorKernels.cu", "addVect");
            _mulKernel = LoadKernel("TensorKernels.ptx", "TensorKernels.cu", "MulVect");
            _tanhKernel = LoadKernel("TensorKernels.ptx", "TensorKernels.cu", "TanhVect");
            _reluKernel = LoadKernel("TensorKernels.ptx", "TensorKernels.cu", "ReLUVect");
            _matmulKernel = LoadKernel("TensorKernels.ptx", "TensorKernels.cu", "MatMulVect");
            _updateKernel = LoadKernel("TensorKernels.ptx", "TensorKernels.cu", "UpdateVect");
        }

        private CudaKernel LoadKernel(string ptxFile, string cuFile, string kernelName)
        {
            var assemblyDir = Path.GetDirectoryName(typeof(GpuBackend).Assembly.Location)!;
            var ptxPath = Path.Combine(assemblyDir, ptxFile);
            var cuPath = Path.Combine(assemblyDir, "Micrograd.Core.Kernels", cuFile);
            
            return File.Exists(ptxPath)
                ? _context.LoadKernel(ptxPath, kernelName)
                : _context.LoadKernel(cuPath, kernelName);
        }

        public Tensor CreateTensor(Shape shape, float[]? data = null)
        {
            var deviceVar = new CudaDeviceVariable<float>(shape.Size);
            
            if (data != null)
            {
                if (data.Length != shape.Size)
                    throw new ArgumentException($"Data length {data.Length} doesn't match shape size {shape.Size}");
                deviceVar.CopyToDevice(data);
            }
            
            return new Tensor(shape, this, deviceVar);
        }

        public Tensor Add(Tensor a, Tensor b)
        {
            if (a.Shape != b.Shape)
                throw new ArgumentException("Tensor shapes must match for addition");

            var result = CreateTensor(a.Shape);
            var aData = (CudaDeviceVariable<float>)a.DeviceData;
            var bData = (CudaDeviceVariable<float>)b.DeviceData;
            var resultData = (CudaDeviceVariable<float>)result.DeviceData;

            var blockSize = 256;
            var gridSize = (a.Size + blockSize - 1) / blockSize;
            
            _addKernel.GridDimensions = gridSize;
            _addKernel.BlockDimensions = blockSize;
            _addKernel.Run(aData.DevicePointer, bData.DevicePointer, resultData.DevicePointer, a.Size);
            _context.Synchronize();

            return result;
        }

        public Tensor Multiply(Tensor a, Tensor b)
        {
            if (a.Shape != b.Shape)
                throw new ArgumentException("Tensor shapes must match for multiplication");

            var result = CreateTensor(a.Shape);
            var aData = (CudaDeviceVariable<float>)a.DeviceData;
            var bData = (CudaDeviceVariable<float>)b.DeviceData;
            var resultData = (CudaDeviceVariable<float>)result.DeviceData;

            var blockSize = 256;
            var gridSize = (a.Size + blockSize - 1) / blockSize;
            
            _mulKernel.GridDimensions = gridSize;
            _mulKernel.BlockDimensions = blockSize;
            _mulKernel.Run(aData.DevicePointer, bData.DevicePointer, resultData.DevicePointer, a.Size);
            _context.Synchronize();

            return result;
        }

        public Tensor Tanh(Tensor input)
        {
            var result = CreateTensor(input.Shape);
            var inputData = (CudaDeviceVariable<float>)input.DeviceData;
            var resultData = (CudaDeviceVariable<float>)result.DeviceData;

            var blockSize = 256;
            var gridSize = (input.Size + blockSize - 1) / blockSize;
            
            _tanhKernel.GridDimensions = gridSize;
            _tanhKernel.BlockDimensions = blockSize;
            _tanhKernel.Run(inputData.DevicePointer, resultData.DevicePointer, input.Size);
            _context.Synchronize();

            return result;
        }

        public Tensor ReLU(Tensor input)
        {
            var result = CreateTensor(input.Shape);
            var inputData = (CudaDeviceVariable<float>)input.DeviceData;
            var resultData = (CudaDeviceVariable<float>)result.DeviceData;

            var blockSize = 256;
            var gridSize = (input.Size + blockSize - 1) / blockSize;
            
            _reluKernel.GridDimensions = gridSize;
            _reluKernel.BlockDimensions = blockSize;
            _reluKernel.Run(inputData.DevicePointer, resultData.DevicePointer, input.Size);
            _context.Synchronize();

            return result;
        }

        public Tensor MatMul(Tensor a, Tensor b)
        {
            if (a.Rank != 2 || b.Rank != 2)
                throw new ArgumentException("MatMul requires 2D tensors");
            
            if (a.Shape[1] != b.Shape[0])
                throw new ArgumentException($"Cannot multiply {a.Shape} x {b.Shape}");

            var M = a.Shape[0];
            var N = b.Shape[1];
            var K = a.Shape[1];
            
            var resultShape = new Shape(M, N);
            var result = CreateTensor(resultShape);
            
            var aData = (CudaDeviceVariable<float>)a.DeviceData;
            var bData = (CudaDeviceVariable<float>)b.DeviceData;
            var resultData = (CudaDeviceVariable<float>)result.DeviceData;

            var blockSize = 16;
            var gridX = (N + blockSize - 1) / blockSize;
            var gridY = (M + blockSize - 1) / blockSize;
            
            _matmulKernel.GridDimensions = new ManagedCuda.VectorTypes.dim3(gridX, gridY, 1);
            _matmulKernel.BlockDimensions = new ManagedCuda.VectorTypes.dim3(blockSize, blockSize, 1);
            _matmulKernel.Run(aData.DevicePointer, bData.DevicePointer, resultData.DevicePointer, M, N, K);
            _context.Synchronize();

            return result;
        }

        public float[] ToHost(Tensor tensor)
        {
            var deviceData = (CudaDeviceVariable<float>)tensor.DeviceData;
            var hostData = new float[tensor.Size];
            deviceData.CopyToHost(hostData);
            return hostData;
        }

        public void UpdateInPlace(Tensor tensor, float[] gradients, float learningRate)
        {
            if (gradients.Length != tensor.Size)
                throw new ArgumentException("Gradient array size must match tensor size");

            using var gradDevice = new CudaDeviceVariable<float>(gradients.Length);
            gradDevice.CopyToDevice(gradients);
            
            var tensorData = (CudaDeviceVariable<float>)tensor.DeviceData;

            var blockSize = 256;
            var gridSize = (tensor.Size + blockSize - 1) / blockSize;
            
            _updateKernel.GridDimensions = gridSize;
            _updateKernel.BlockDimensions = blockSize;
            _updateKernel.Run(tensorData.DevicePointer, gradDevice.DevicePointer, learningRate, tensor.Size);
            _context.Synchronize();
        }

        public void Dispose()
        {
            _context?.Dispose();
        }
    }
}