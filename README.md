# micrograd.net

because why not port [karpathy's micrograd](https://github.com/karpathy/micrograd) to C# 🤷‍♂️

## what is this

tiny neural network lib with automatic differentiation. builds computational graphs, does backprop, trains networks. the usual.

## usage

```bash
dotnet run --project Micrograd.Examples
```

watch it learn XOR or whatever. there's also tests if you're into that:

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

## why C#

¯\\_(ツ)_/¯

someone had to do it. plus operator overloading is actually pretty nice for this stuff.

## structure

- `Micrograd.Core/` - the actual library (like 300 lines)
- `Micrograd.Tests/` - 37 tests that all pass somehow
- `Micrograd.Examples/` - demos including XOR that gets 100% accuracy

## notes

- it works
- probably don't use this for anything important
- pure C#, no dependencies
- educational purposes and/or mild entertainment

that's it. 