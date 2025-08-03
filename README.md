# micrograd.net

because why not port [karpathy's micrograd](https://github.com/karpathy/micrograd) to C# 🤷‍♂️

## what is this

tiny neural network lib with automatic differentiation. builds computational graphs, does backprop, trains networks. the usual.

**now with GPU acceleration** 🔥 - because your NVIDIA GPU was getting lonely.

## usage

basic examples (CPU + GPU):
```bash
dotnet run --project Micrograd.Examples
```

watch it learn XOR or whatever. also shows your GPU absolutely destroying CPU performance.

quick GPU benchmark (recommended):
```bash
dotnet run --project Micrograd.Examples gpu
```

heavy GPU stress test:
```bash
dotnet run --project Micrograd.Examples benchmark
```

there's also tests if you're into that:
```bash
dotnet test
```

## examples

basic autodiff:
```csharp
var x = new Value(2.0);
var y = x * 2 + 1;
y.Backward();
Console.WriteLine(x.Grad); // 2.0
```

neural network:
```csharp
var mlp = new MLP(2, new[] { 4, 1 });
var prediction = mlp.ForwardSingle(new[] { new Value(1.0), new Value(-1.0) });
```

GPU tensor operations:
```csharp
var gpu = new GpuBackend(); // your NVIDIA GPU
var a = gpu.CreateTensor(new Shape(2048, 2048), data);
var b = gpu.CreateTensor(new Shape(2048, 2048), moreData);
var result = gpu.MatMul(a, b); // blazing fast on GPU
```

## performance

recent benchmarks on NVIDIA A10:
- 512×512 matrix: **425x faster** than CPU
- 1024×1024 matrix: **1,922x faster** than CPU  
- 2048×2048 matrix: **2,551x faster** than CPU

your GPU probably isn't bored anymore.

## why C#

¯\\_(ツ)_/¯

someone had to do it. plus operator overloading is actually pretty nice for this stuff.

## structure

- `Micrograd.Core/` - the actual library (now with GPU backend)
- `Micrograd.Tests/` - tests that all pass somehow
- `Micrograd.Examples/` - demos including XOR that gets 100% accuracy
  - `Program.cs` - main examples (CPU Value-based + GPU Tensor-based)
  - `TensorProgram.cs` - GPU tensor operations with automatic CPU fallback
  - `GpuBenchmark.cs` - heavy GPU performance testing
  - `QuickGpuTest.cs` - quick matrix multiplication benchmarks

## requirements

- .NET 9.0
- NVIDIA GPU + CUDA drivers (for GPU acceleration)
- falls back to CPU if no GPU available

## notes

- it works
- GPU acceleration actually works really well
- probably still don't use this for anything important
- ManagedCuda dependency for the GPU magic
- educational purposes and/or mild entertainment and/or showing off your GPU

that's it. 