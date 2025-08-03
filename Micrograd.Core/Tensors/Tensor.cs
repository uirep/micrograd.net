using System;
using ManagedCuda;

namespace Micrograd.Core
{
    public readonly struct Shape : IEquatable<Shape>
    {
        public readonly int[] Dims;
        public readonly int Rank => Dims?.Length ?? 0;
        public readonly int Size => Rank == 0 ? 1 : GetSize();

        public Shape(params int[] dims)
        {
            if (dims == null || dims.Length == 0)
                throw new ArgumentException("Shape must have at least one dimension");
            
            for (int i = 0; i < dims.Length; i++)
            {
                if (dims[i] <= 0)
                    throw new ArgumentException($"Dimension {i} must be positive, got {dims[i]}");
            }
            
            Dims = new int[dims.Length];
            Array.Copy(dims, Dims, dims.Length);
        }

        private int GetSize()
        {
            int size = 1;
            for (int i = 0; i < Dims.Length; i++)
                size *= Dims[i];
            return size;
        }

        public int this[int index] => Dims[index];

        public bool Equals(Shape other)
        {
            if (Rank != other.Rank) return false;
            for (int i = 0; i < Rank; i++)
                if (Dims[i] != other.Dims[i]) return false;
            return true;
        }

        public override bool Equals(object? obj) => obj is Shape shape && Equals(shape);
        public override int GetHashCode()
        {
            var hash = 17;
            for (int i = 0; i < Rank; i++)
                hash = hash * 31 + Dims[i];
            return hash;
        }

        public static bool operator ==(Shape left, Shape right) => left.Equals(right);
        public static bool operator !=(Shape left, Shape right) => !left.Equals(right);

        public override string ToString() => $"[{string.Join(", ", Dims)}]";
    }

    public interface ITensorBackend : IDisposable
    {
        Tensor CreateTensor(Shape shape, float[]? data = null);
        Tensor MatMul(Tensor a, Tensor b);
        Tensor Add(Tensor a, Tensor b);
        Tensor Multiply(Tensor a, Tensor b);
        Tensor Tanh(Tensor input);
        Tensor ReLU(Tensor input);
        float[] ToHost(Tensor tensor);
        void UpdateInPlace(Tensor tensor, float[] gradients, float learningRate);
    }

    public readonly struct Tensor : IDisposable
    {
        public readonly Shape Shape;
        public readonly ITensorBackend Backend;
        public readonly object DeviceData;
        public readonly int Id;

        private static int _nextId = 0;

        internal Tensor(Shape shape, ITensorBackend backend, object deviceData)
        {
            Shape = shape;
            Backend = backend;
            DeviceData = deviceData;
            Id = System.Threading.Interlocked.Increment(ref _nextId);
        }

        public int Size => Shape.Size;
        public int Rank => Shape.Rank;

        public static Tensor operator +(Tensor a, Tensor b) => a.Backend.Add(a, b);
        public static Tensor operator *(Tensor a, Tensor b) => a.Backend.Multiply(a, b);

        public Tensor MatMul(Tensor other) => Backend.MatMul(this, other);
        public Tensor Tanh() => Backend.Tanh(this);
        public Tensor ReLU() => Backend.ReLU(this);
        public float[] ToHost() => Backend.ToHost(this);

        public void Dispose()
        {
            if (DeviceData is IDisposable disposable)
                disposable.Dispose();
        }

        public override string ToString() => $"Tensor{Shape}[{Id}]";
    }
}