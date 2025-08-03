using System;

namespace Micrograd.Core.Backends
{
    public class CpuBackend : ITensorBackend
    {
        private class CpuTensorData
        {
            public float[] Data { get; }
            
            public CpuTensorData(int size)
            {
                Data = new float[size];
            }
            
            public CpuTensorData(float[] data)
            {
                Data = new float[data.Length];
                Array.Copy(data, Data, data.Length);
            }
        }

        public Tensor CreateTensor(Shape shape, float[]? data = null)
        {
            var tensorData = data != null 
                ? new CpuTensorData(data) 
                : new CpuTensorData(shape.Size);
            
            return new Tensor(shape, this, tensorData);
        }

        public Tensor Add(Tensor a, Tensor b)
        {
            if (a.Shape != b.Shape)
                throw new ArgumentException("Tensor shapes must match for addition");

            var aData = ((CpuTensorData)a.DeviceData).Data;
            var bData = ((CpuTensorData)b.DeviceData).Data;
            var result = new float[a.Size];
            
            for (int i = 0; i < a.Size; i++)
                result[i] = aData[i] + bData[i];
            
            return CreateTensor(a.Shape, result);
        }

        public Tensor Multiply(Tensor a, Tensor b)
        {
            if (a.Shape != b.Shape)
                throw new ArgumentException("Tensor shapes must match for multiplication");

            var aData = ((CpuTensorData)a.DeviceData).Data;
            var bData = ((CpuTensorData)b.DeviceData).Data;
            var result = new float[a.Size];
            
            for (int i = 0; i < a.Size; i++)
                result[i] = aData[i] * bData[i];
            
            return CreateTensor(a.Shape, result);
        }

        public Tensor Tanh(Tensor input)
        {
            var inputData = ((CpuTensorData)input.DeviceData).Data;
            var result = new float[input.Size];
            
            for (int i = 0; i < input.Size; i++)
                result[i] = (float)Math.Tanh(inputData[i]);
            
            return CreateTensor(input.Shape, result);
        }

        public Tensor ReLU(Tensor input)
        {
            var inputData = ((CpuTensorData)input.DeviceData).Data;
            var result = new float[input.Size];
            
            for (int i = 0; i < input.Size; i++)
                result[i] = Math.Max(0f, inputData[i]);
            
            return CreateTensor(input.Shape, result);
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
            
            var aData = ((CpuTensorData)a.DeviceData).Data;
            var bData = ((CpuTensorData)b.DeviceData).Data;
            var result = new float[M * N];
            
            for (int row = 0; row < M; row++)
            {
                for (int col = 0; col < N; col++)
                {
                    float sum = 0f;
                    for (int k = 0; k < K; k++)
                    {
                        sum += aData[row * K + k] * bData[k * N + col];
                    }
                    result[row * N + col] = sum;
                }
            }
            
            return CreateTensor(new Shape(M, N), result);
        }

        public float[] ToHost(Tensor tensor)
        {
            var data = ((CpuTensorData)tensor.DeviceData).Data;
            var result = new float[data.Length];
            Array.Copy(data, result, data.Length);
            return result;
        }

        public void UpdateInPlace(Tensor tensor, float[] gradients, float learningRate)
        {
            if (gradients.Length != tensor.Size)
                throw new ArgumentException("Gradient array size must match tensor size");

            var data = ((CpuTensorData)tensor.DeviceData).Data;
            
            for (int i = 0; i < data.Length; i++)
                data[i] -= learningRate * gradients[i];
        }

        public void Dispose()
        {
        }
    }
}