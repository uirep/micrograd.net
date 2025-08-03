using Xunit;
using Micrograd.Core;
using System.Linq;

namespace Micrograd.Tests.Core
{
    public class LossFunctionTests
    {
        [Fact]
        public void MSE_SingleValue()
        {
            var prediction = new Value(2.0);
            var target = new Value(1.0);
            
            var loss = LossFunctions.MeanSquaredError(prediction, target);
            
            Assert.Equal(1.0, loss.Data, 1e-10);
        }

        [Fact]
        public void MSE_MultipleValues()
        {
            var predictions = new[] { new Value(2.0), new Value(3.0) };
            var targets = new[] { new Value(1.0), new Value(2.0) };
            
            var loss = LossFunctions.MeanSquaredError(predictions, targets);
            
            Assert.Equal(1.0, loss.Data, 1e-10);
        }

        [Fact]
        public void MSE_BackwardPass()
        {
            var prediction = new Value(2.0);
            var target = new Value(1.0);
            
            var loss = LossFunctions.MeanSquaredError(prediction, target);
            loss.Backward();
            
            Assert.Equal(2.0, prediction.Grad, 1e-10);
            Assert.Equal(-2.0, target.Grad, 1e-10);
        }

        [Fact]
        public void SGDOptimizer_ParameterUpdate()
        {
            var param = new Value(1.0);
            param.Grad = 0.5;
            
            var optimizer = new SGDOptimizer(0.1);
            optimizer.Step(new[] { param });
            
            Assert.Equal(0.95, param.Data, 1e-10);
        }

        [Fact]
        public void SGDOptimizer_MultipleParameters()
        {
            var param1 = new Value(1.0);
            var param2 = new Value(2.0);
            param1.Grad = 0.5;
            param2.Grad = -0.3;
            
            var optimizer = new SGDOptimizer(0.1);
            optimizer.Step(new[] { param1, param2 });
            
            Assert.Equal(0.95, param1.Data, 1e-10);
            Assert.Equal(2.03, param2.Data, 1e-10);
        }
    }
}