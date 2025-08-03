using Xunit;
using Micrograd.Core;
using Micrograd.Core.Backends;
using System.Linq;

namespace Micrograd.Tests.Tensors
{
    public class TensorNeuralNetworkTests : IDisposable
    {
        private readonly ITensorBackend _backend;

        public TensorNeuralNetworkTests()
        {
            try
            {
                _backend = new GpuBackend();
            }
            catch
            {
                _backend = new CpuBackend();
            }
        }

        [Fact]
        public void TensorNeuron_CreatesCorrectly()
        {
            var neuron = new TensorNeuron(3, true, _backend);
            
            Assert.Equal(3, neuron.Weights.Count);
            Assert.NotNull(neuron.Bias);
            Assert.True(neuron.NonLinear);
            
            neuron.Dispose();
        }

        [Fact]
        public void TensorNeuron_Forward()
        {
            var neuron = new TensorNeuron(2, false, _backend);
            
            var inputs = new[]
            {
                new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 1.0f })),
                new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 2.0f }))
            };
            
            var output = neuron.Forward(inputs);
            Assert.NotNull(output);
            
            foreach (var input in inputs)
                input.Dispose();
            output.Dispose();
            neuron.Dispose();
        }

        [Fact]
        public void TensorLayer_CreatesCorrectly()
        {
            var layer = new TensorLayer(3, 4, true, _backend);
            
            Assert.Equal(4, layer.Neurons.Count);
            Assert.Equal(16, layer.Parameters().Count());
            
            layer.Dispose();
        }

        [Fact]
        public void TensorLayer_Forward()
        {
            var layer = new TensorLayer(2, 3, false, _backend);
            
            var inputs = new[]
            {
                new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 0.5f })),
                new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { -0.3f }))
            };
            
            var outputs = layer.Forward(inputs);
            Assert.Equal(3, outputs.Count);
            
            foreach (var input in inputs)
                input.Dispose();
            foreach (var output in outputs)
                output.Dispose();
            layer.Dispose();
        }

        [Fact]
        public void TensorMLP_CreatesCorrectly()
        {
            var mlp = new TensorMLP(2, new[] { 4, 3, 1 }, _backend);
            
            Assert.Equal(3, mlp.Layers.Count);
            Assert.Equal(26, mlp.Parameters().Count());
            
            mlp.Dispose();
        }

        [Fact]
        public void TensorMLP_Forward()
        {
            var mlp = new TensorMLP(2, new[] { 3, 1 }, _backend);
            
            var inputs = new[]
            {
                new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 1.0f })),
                new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { -1.0f }))
            };
            
            var output = mlp.ForwardSingle(inputs);
            Assert.NotNull(output);
            
            foreach (var input in inputs)
                input.Dispose();
            output.Dispose();
            mlp.Dispose();
        }

        [Fact]
        public void TensorMLP_TrainingSingleStep()
        {
            var mlp = new TensorMLP(2, new[] { 3, 1 }, _backend);
            var optimizer = new TensorSGDOptimizer(0.01f);
            
            var inputs = new[]
            {
                new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 1.0f })),
                new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 0.0f }))
            };
            var target = new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 1.0f }));
            
            var prediction = mlp.ForwardSingle(inputs);
            var loss = TensorLossFunctions.MeanSquaredError(prediction, target);
            
            mlp.ZeroGrad();
            loss.Backward();
            optimizer.Step(mlp.Parameters());
            
            Assert.True(loss.Data.ToHost()[0] >= 0);
            
            foreach (var input in inputs)
                input.Dispose();
            target.Dispose();
            prediction.Dispose();
            loss.Dispose();
            mlp.Dispose();
        }

        [Fact]
        public void TensorMLP_XORLearning()
        {
            var mlp = new TensorMLP(2, new[] { 4, 1 }, _backend);
            var optimizer = new TensorSGDOptimizer(0.1f);
            
            var xorData = new[]
            {
                (new[] { 0.0f, 0.0f }, 0.0f),
                (new[] { 0.0f, 1.0f }, 1.0f),
                (new[] { 1.0f, 0.0f }, 1.0f),
                (new[] { 1.0f, 1.0f }, 0.0f)
            };
            
            var initialLoss = 0.0f;
            var finalLoss = 0.0f;
            
            for (int epoch = 0; epoch < 50; epoch++)
            {
                var epochLoss = 0.0f;
                
                foreach (var (inputData, targetData) in xorData)
                {
                    var inputs = inputData.Select(x => new TensorValue(_backend.CreateTensor(new Shape(1), new[] { x }))).ToArray();
                    var target = new TensorValue(_backend.CreateTensor(new Shape(1), new[] { targetData }));
                    
                    var prediction = mlp.ForwardSingle(inputs);
                    var loss = TensorLossFunctions.MeanSquaredError(prediction, target);
                    
                    mlp.ZeroGrad();
                    loss.Backward();
                    optimizer.Step(mlp.Parameters());
                    
                    epochLoss += loss.Data.ToHost()[0];
                    
                    foreach (var input in inputs)
                        input.Dispose();
                    target.Dispose();
                    prediction.Dispose();
                    loss.Dispose();
                }
                
                if (epoch == 0) initialLoss = epochLoss;
                if (epoch == 49) finalLoss = epochLoss;
            }
            
            Assert.True(finalLoss < initialLoss, $"Loss should decrease: {initialLoss} -> {finalLoss}");
            
            mlp.Dispose();
        }

        [Fact]
        public void TensorLossFunctions_MSE()
        {
            var pred = new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 2.0f }));
            var target = new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 1.0f }));
            
            var loss = TensorLossFunctions.MeanSquaredError(pred, target);
            
            Assert.Equal(1.0f, loss.Data.ToHost()[0], 1e-6f);
            
            pred.Dispose();
            target.Dispose();
            loss.Dispose();
        }

        [Fact]
        public void TensorSGDOptimizer_ParameterUpdate()
        {
            var param = new TensorValue(_backend.CreateTensor(new Shape(1), new float[] { 1.0f }));
            param.Grad = _backend.CreateTensor(new Shape(1), new float[] { 0.5f });
            
            var optimizer = new TensorSGDOptimizer(0.1f);
            var originalValue = param.Data.ToHost()[0];
            
            optimizer.Step(new[] { param });
            
            var newValue = param.Data.ToHost()[0];
            Assert.Equal(0.95f, newValue, 1e-6f);
            
            param.Dispose();
        }

        public void Dispose()
        {
            _backend?.Dispose();
        }
    }
}