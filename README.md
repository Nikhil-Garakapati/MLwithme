# Neural Networks with Python
###  What does a basic neural network contains?
- Input layer, Hidden layers and an output layer
- Activation funtion, loss function
- Weights and Bias

Let's build a 2 layered neural network( Exclude the input layer when you're couting the number of layers ). A Neural network follows feed-forward and backpropagation algorithm. In neural network, each node is connected to the another node in the next layer and it is represented by weights. Biologically, those weights are synapses. 

A simple 2 layered neural network looks like this

<p align="center">
  <img src="">
</p>

Creating a neural network
```python
Class NeuralNetwork:
    def __init__(self,x,y):
        self.input    = x
        self.y        = y
        self.weights1 = np.random.rand(self.input.shape[1],4)
        seld.weights2 = np.random.rand(4,1)
        self.output   = np.zeros(y.shape)  

```
As the network is Feed forwarded
```python
def feedfoward(self):
    self.layer1 = sigmoid(np.dot(self.input, self.weights1))
    self.output = sigmoid(np.dot(self.layer1, self.weights2))
```
The above lines are matrix multiplication of the input node values and weights
of the hidden layer and output layer.

There are many activation functions that can be used in neural nets. Let's get started with **Sigmoid(σ)** activation function. This neural network gives the prediction either **1 or 0**. The sigmoid function typically looks like

<p align="center">
  <img src="">
</p>

This sigmoid function is an logisitic funtion that's applied in hidden layers and in output layers. The reason we use activation functions is to apply non-linear activity to the model. The model gets complexed. And the most important thing we need to remember is **Activation function should be differentiable**. 

### What happens if we don't use activation function?
If we don't apply the activation function, then it would be a simple linear model. The complexity of the model will be limited. The model will have less power to learn data. It would just like a normal Regression model which has limited power and does not performs well most of the times. If we want to deal with images, voice, speech etc., we have to use activation functions to complex our model.
Now let's back to our sigmoid activation function. The range of sigmoid lies between 0 and 1. The graph of sigmoid function looks like a S-Shaped curve. So, it is also called as **S-Fucntion**.

- If x approaches -∞, the value is 0
- If x approaches +∞, the value is 1

```python
def sigmoid(x):
    return 1.0/(1+np.exp(-x))
```

Let the expected output be **1** and denoted by **ȳ** and obtained output be **0.7**. So there's an error of **0.3** which is to be corrected. For this, we have to tune the weights of the nodes in order to minimize the error. So, we have backpropagate to tune the weights.

In order to backpropagate, we have the derivate the loss function with respect to weights and bias. There are many loss functions out there that can be used. We use **SSE-Loss function (Sum of squared errors)**. 

<p align="center">
  <img src="">
</p>

Now, we have the derivate the SSE-loss function.

<p align="center">
  <img src="">
</p>

This loss function is applied to every layer and when it comes to code it looks like 
```python
def backpropagation(self):
    d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) *                         sigmoid_derivative(self.output)))
    d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) *                  sigmoid_derivative(self.output), self.weights2.T) *                        sigmoid_derivative(self.layer1)))
#Update the weights
self.weights1 = d_weights1 + 1
self.weights2 = d_weights2 + 1
    
def sigmoid_derivative(x):
    return x * (1.0 - x)

```

That's it, we have made it to the end. Now let's give the input and output to the neural network and run it 
```python
if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)
```

When the number of iterations increases, the minimization of error increases.
Here we use the term epochs for iterations. ( But number of Iterations are not equal to number of epochs. It depends upon the size of the dataset )

```python
epochs=1000
for i in range(epochs):
    nn.feedforward()
    nn.backpropagation()

print(nn.output)
    
```
#### There's still much to learn about the neural nets in this. For example
- What are the other activation functions other than sigmoid?
- Which activation function shoudl be used?.
- Using a learning rate when training the net.









