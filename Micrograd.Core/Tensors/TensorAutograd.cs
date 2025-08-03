using System;
using System.Collections.Generic;
using System.Linq;

namespace Micrograd.Core
{
    public class TensorValue : IDisposable
    {
        public Tensor Data { get; set; }
        public Tensor Grad { get; set; }
        public string Op { get; private set; }
        public string Label { get; set; }
        
        private readonly HashSet<TensorValue> _prev;
        private Action _backward;

        public TensorValue(Tensor data, IEnumerable<TensorValue>? children = null, string op = "", string label = "")
        {
            Data = data;
            Op = op;
            Label = label;
            _prev = children?.ToHashSet() ?? new HashSet<TensorValue>();
            _backward = () => { };
        }

        public static TensorValue operator +(TensorValue a, TensorValue b)
        {
            var result = new TensorValue(a.Data + b.Data, new[] { a, b }, "+");
            
            result._backward = () =>
            {
                if (a.Grad.DeviceData == null)
                    a.Grad = a.Data.Backend.CreateTensor(a.Data.Shape, new float[a.Data.Size]);
                if (b.Grad.DeviceData == null)
                    b.Grad = b.Data.Backend.CreateTensor(b.Data.Shape, new float[b.Data.Size]);
                
                a.Grad = a.Grad + result.Grad;
                b.Grad = b.Grad + result.Grad;
            };
            
            return result;
        }

        public static TensorValue operator *(TensorValue a, TensorValue b)
        {
            var result = new TensorValue(a.Data * b.Data, new[] { a, b }, "*");
            
            result._backward = () =>
            {
                if (a.Grad.DeviceData == null)
                    a.Grad = a.Data.Backend.CreateTensor(a.Data.Shape, new float[a.Data.Size]);
                if (b.Grad.DeviceData == null)
                    b.Grad = b.Data.Backend.CreateTensor(b.Data.Shape, new float[b.Data.Size]);
                
                a.Grad = a.Grad + (b.Data * result.Grad);
                b.Grad = b.Grad + (a.Data * result.Grad);
            };
            
            return result;
        }

        public TensorValue MatMul(TensorValue other)
        {
            var result = new TensorValue(Data.MatMul(other.Data), new[] { this, other }, "matmul");
            
            result._backward = () =>
            {
                if (Grad.DeviceData == null)
                    Grad = Data.Backend.CreateTensor(Data.Shape, new float[Data.Size]);
                if (other.Grad.DeviceData == null)
                    other.Grad = other.Data.Backend.CreateTensor(other.Data.Shape, new float[other.Data.Size]);
                
                var transposeA = CreateTranspose(other.Data);
                var transposeB = CreateTranspose(Data);
                
                Grad = Grad + result.Grad.MatMul(transposeA);
                other.Grad = other.Grad + transposeB.MatMul(result.Grad);
            };
            
            return result;
        }

        private Tensor CreateTranspose(Tensor tensor)
        {
            if (tensor.Rank != 2)
                throw new ArgumentException("Transpose only supports 2D tensors");
            
            var hostData = tensor.ToHost();
            var M = tensor.Shape[0];
            var N = tensor.Shape[1];
            var transposed = new float[M * N];
            
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    transposed[j * M + i] = hostData[i * N + j];
                }
            }
            
            return tensor.Backend.CreateTensor(new Shape(N, M), transposed);
        }

        public TensorValue Tanh()
        {
            var result = new TensorValue(Data.Tanh(), new[] { this }, "tanh");
            
            result._backward = () =>
            {
                if (Grad.DeviceData == null)
                    Grad = Data.Backend.CreateTensor(Data.Shape, new float[Data.Size]);
                
                var ones = Data.Backend.CreateTensor(Data.Shape, Enumerable.Repeat(1f, Data.Size).ToArray());
                var tanhSquared = result.Data * result.Data;
                var derivative = ones + (tanhSquared * Data.Backend.CreateTensor(Data.Shape, Enumerable.Repeat(-1f, Data.Size).ToArray()));
                
                Grad = Grad + (derivative * result.Grad);
            };
            
            return result;
        }

        public TensorValue ReLU()
        {
            var result = new TensorValue(Data.ReLU(), new[] { this }, "relu");
            
            result._backward = () =>
            {
                if (Grad.DeviceData == null)
                    Grad = Data.Backend.CreateTensor(Data.Shape, new float[Data.Size]);
                
                var hostData = Data.ToHost();
                var mask = hostData.Select(x => x > 0 ? 1f : 0f).ToArray();
                var maskTensor = Data.Backend.CreateTensor(Data.Shape, mask);
                
                Grad = Grad + (maskTensor * result.Grad);
            };
            
            return result;
        }

        public void Backward()
        {
            var topo = new List<TensorValue>();
            var visited = new HashSet<TensorValue>();
            
            void BuildTopo(TensorValue v)
            {
                if (!visited.Contains(v))
                {
                    visited.Add(v);
                    foreach (var child in v._prev)
                        BuildTopo(child);
                    topo.Add(v);
                }
            }
            
            BuildTopo(this);
            
            Grad = Data.Backend.CreateTensor(Data.Shape, Enumerable.Repeat(1f, Data.Size).ToArray());
            
            for (int i = topo.Count - 1; i >= 0; i--)
                topo[i]._backward();
        }

        public void ZeroGrad()
        {
            var visited = new HashSet<TensorValue>();
            
            void ZeroGradRecursive(TensorValue v)
            {
                if (!visited.Contains(v))
                {
                    visited.Add(v);
                    if (v.Grad.DeviceData != null)
                    {
                        v.Grad.Dispose();
                        v.Grad = v.Data.Backend.CreateTensor(v.Data.Shape, new float[v.Data.Size]);
                    }
                    foreach (var child in v._prev)
                        ZeroGradRecursive(child);
                }
            }
            
            ZeroGradRecursive(this);
        }

        public void Dispose()
        {
            Data.Dispose();
            if (Grad.DeviceData != null)
                Grad.Dispose();
        }

        public override string ToString()
        {
            var label = !string.IsNullOrEmpty(Label) ? $"{Label}:" : "";
            return $"TensorValue({label}{Data})";
        }
    }
}