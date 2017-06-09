from EvoNN import NeuralNet 
import random

'''
Note: Built on python 3.6.1 

Here is an example of how you coul use the neural net.  In this example
we will be setting up the neural net for a mock world. In this world 
creatures will be able to know their speed, what they see, how fast
they are turning, and some current utility value.  With these value 
the creature should be able to respond to its environment. The creature
will also be able to procreate.  Any output that is labeled identicle to
the input will automatically have its values transfered between cycles
'''


def main():

	#initialize the neural net
	NN = NeuralNet()

	#Set the IO neurons

	#adding input neurons to the neural net
	#Syntax:
	#label -> the name of the input
	#duplicate -> number of neurons associated with the input
	#min -> minimum amount of times the same input is allowed 
	#max -> maximum amount of times the same input is allowed 
	#append as follows:	[label, duplicate, [min,max]]
	#appending this would result in only one input:  [label, duplicate, []]
	NN.inputs.append(['speed', 2, []])			# there will be 1 'speed' input, comprised of 2 neruons [x,y]
	NN.inputs.append(['eyes', 3, [1,3]])		# there will be between 1 and 3 'eyes' inputs, each comprised of 3 neurons [R,G,B]
	NN.inputs.append(['rotation speed', 1, []]) # there will be 1 'rotation speed' input, comprised of 1 neuron

	#adding output neurons to the neural net
	NN.outputs.append(['speed', 2, []])			# there will be 1 'speed' output, comprised of 2 neurons [x,y]
	NN.outputs.append(['rotation speed', 1, []]) # there will be 1 'rotation speed' output, comprised of 1 neuron
	NN.outputs.append(['procreate', 1, []])		# there will be 1 'procreate' output, comprised of 1 neuron

	#define the hidden layer structure
	#Syntax:
	#min -> minimum number of neurons per hidden layer
	#max -> maximum nuber of neurons per hidden layer
	#layers -> number of hidden layers
	#set as follows: [[min,max],layers]
	#if you set layers > 0, you will have layers independantly connected with [min,max] range of neurons per layer
	#if you set layers = 0, wou will have one layer all inteconnected with [min,max] range of neurons in the layer
	#play with this number and see how it responds in NN.show()
	NN.hidden = [[5,10],3]

	#develops the neural net from the inputs
	NN.build()

	#set mutation variables, they are set high to show it functioning
	#larger the variability, the large weights change during mutations
	variablilty = .75 #no max, should be <= 1
	#larger the weight_change_prob the higher the chance is of any given weight being altered
	weight_change_prob = .5 #max 1
	#larger the connection_change_prob the higher the chance of a connection being deleted or added
	connection_change_prob = .5 #max 1
	#larger the neuron_change_prob the higher the chance a neuron can be added or removed
	neuron_change_prob = .05 #max 1

	#think_cycles are important if your NN is is interconnected (ie. you set layers = 0)
	#this is because some neurons may have inputs that are effected by its output.  So 
	#additional think cycles will help the values converge
	think_cycles = 1

	#This loop will quickly generate mutations and show its effect on the nerual net
	#green line is a positive weight, with its brightness corresponding to the magnitude
	#red line is a negative weight, with its brightness corresponding to the magnitude
	#grey circles are neurons whos value is shown through its brightness

	for i in range(0,50):
		#print(i)
		print(NN.available)
		NN.think(think_cycles)
		NN.show()

		#This is just showing how you can modify NN inputs during its run
		#Syntax:
		#label = which neuron do you want to edit
		#num = how many of the input you want to add or remove
		#	Note: this value is hard capped at the min/max value you specified when first initializing the input
		if random.random() <= .5:
			NN.change_input('eyes',1)
		if random.random() <= .5:
			NN.change_input('eyes',-1)
		#mutating the 
		NN.mutate(variablilty, weight_change_prob, connection_change_prob, neuron_change_prob)
	#after all the mutations have been made, it will print the outputs as such:
	#[[output1, value1],[output2, value2],...,[outputn, valuen]]
	print(NN.get_outputs())
	return

main()