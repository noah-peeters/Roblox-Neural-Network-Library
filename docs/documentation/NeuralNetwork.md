# **Abstract NeuralNetwork Class**
This class is responsible for the majority of the functionality behind all present neural networks. Though it does not construct anything, it does manage most of what a neural network is. <br>
This class is abstract and thus cannot be used directly.

<div class=functionDoc>
abstract NeuralNetwork .new(dictionary customSettings)
Creates and returns the NeuralNetwork according to the given settings as !customSettings!. The settings determine how the network will perform and function and is open to customization.
The available setting parameters and their default values are below.

```lua
local customSettings {
    Optimizer = StochasticGradientDescent.new();
	SoftMax = false;
}
```
</div>

<div class=functionDoc>
NeuralNetwork .newFromSave(string serial)
Deserializes and returns the neural network from the !serial! string: the save string.
</div>

<div class=functionDoc>
dictionary :(dictionary inputValues, bool doNotClearOtherInputValues=false)
Propagates the network with the given !inputValues! if provided. If not, the network will run with the input values already set previously. If !doNotClearOtherInputValues! is true, any inputs that are missing from the !inputValues! dictionary are set to 0.
<br><br>
This function is fired when the NeuralNetwork object is called, such as:

```lua
neuralNetwork:({Input1 = 0.5, Input2 = -0.2})
```
</div>

<div class=functionDoc>
string :Save()
Returns the serialized form of the neural network as a string, allowing you to save the network.
</div>

<div class=functionDoc>
void :ConnectNodes(Node inNode, Node outNode, bool checkOveride=false)
Creates a synapse for the 2 nodes with !inNode! as the input and !outNode! as the output. If !checkOveride! is true, this function will not check if the synapse between these 2 nodes already exists; this is purely for internal performance purposes where the check is unnecessary. Do not set this to true if you are experimenting unless you're confident.
</div>

<div class=functionDoc>
dictionary :GetOutputValues()
Returns the current output values.
</div>

<div class=functionDoc>
void :SetInputValues(dictionary inputValues, bool doNotClearOtherInputValues=false)
Sets the given input values as !inputValues! into the input nodes. If !doNotClearOtherInputValues! is true, any inputs that are missing from the !inputValues! dictionary are set to 0.
</div>

<div class=functionDoc>
void :ClearInputValues()
Sets all input values of the input nodes to 0.
</div>

<div class=functionDoc>
void :ClearValues()
Sets all the output values of every functional node to 0.
</div>

<div class=functionDoc>
array :GetNodes()
Returns every node in the network in an unreliable order.
</div>

<div class=functionDoc>
array :GetInputNodes()
Returns the input nodes in the network in the order they were assigned by the user upon network creation.
</div>

<div class=functionDoc>
void :AddInputNode(InputNode inputNode)
Adds input node !inputNode! to the network.
</div>

<div class=functionDoc>
array :GetOutputNodes()
Returns the output nodes in the network in the order they were assigned by the user upon network creation.
</div>

<div class=functionDoc>
void :AddOutputNode(OutputNode outputNode)
Adds output node !outputNode! to the network.
</div>

<div class=functionDoc>
array :GetHiddenNodes()
Returns the hidden nodes in the network in the order they were created (first node to last node of first layer, first node to last node of second layer, etc).
</div>

<div class=functionDoc>
void :AddHiddenNode(Node node)
Adds the hidden node !node! to the network.
</div>

<div class=functionDoc>
array :GetFunctionNodes()
Returns the functional nodes in the network. Functional nodes, in my definition, are nodes that have a working bias and weight set. Thus, hidden nodes and output nodes are all functional nodes.
</div>

<div class=functionDoc>
void :AddNode(Node node)
Adds the node !node! to the network.
</div>

<div class=functionDoc>
BackPropagator :GetBackPropagator()
Returns the BackPropagator object for this network.
</div>

<div class=functionDoc>
void :RandomizeWeights(number min=-0.5, number max=0.5)
Randomizes all weights in the network with the given minimum !min! and maximum !max! values.
</div>

<div class=functionDoc>
Optimizer :GetOptimizer()
Returns the Optimizer object the network is using.
</div>

<div class=functionDoc>
void :AddRandomNoise(number min, number max)
Adds a random noise to every parameter in the network with the given minimum !min! and maximum !max! values.
</div>