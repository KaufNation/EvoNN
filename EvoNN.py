'''
Developed by: Joshua Kaufman
License: Apache License 2.0
Note: 
	Built on python 3.6.1
	All functions that use ranges are inclusive.  So a range [5,10] can be [5,6,7,8,9,10]
	The way the algorithm is built is to increase the chance of removal of neurons and connections. 
		This wont effect outcomes but will help to create lightweight solutions
Enjoy
'''
import sys, pygame
import random
import math
import copy
import io

def sigmoid(x):
	#Limiting x value to mitigate overflow. Results will be off by up to .000000002
	if x > 20:
		x = 20
	if x < -20:
		x = -20
	return 1 / (1 + math.exp(-x))

class Neuron:
	def __init__(self):
		#Inbound neuronal connections
		self.inbound = []
		self.label = None
		self.cur_value = 0.5
		#Value before sigmoid
		self.raw_next_value = 0.0
		self.pos = [0,0]
		return

class NeuralNet:
	def __init__(self):
		self.inputs = []
		self.hidden = []
		self.raw_hidden = []
		self.outputs = []
		self.neuron_list = []
		self.available = []

		self.num_active_i_neurons = 0
		self.num_active_h_neurons = []
		self.num_active_o_neurons = 0

		self.num_pos_i_neurons = 0
		self.num_pos_h_neurons = []
		self.num_pos_o_neurons = 0

		self.num_active_neurons = 0
		self.num_pos_neurons = 0

		self.num_connections = 0

	def build(self):
		hidden = []
		#determine number of pos and active input neurons. Then update neuron_list
		self.inputs, self.num_pos_i_neurons, self.num_active_i_neurons = self.io_extrapolator(self.inputs)
		#print(self.inputs, self.num_active_i_neurons, self.num_pos_i_neurons)
		#determine number of pos and active hidden neurons. Then update neuron_list
		sample_list = list(range(0, self.hidden[0][1]))
		num_h_layers = self.hidden[1]
		if num_h_layers == 0:
			num_h_layers = 1
		for i in range(0,num_h_layers):
			sample_size = random.randint(self.hidden[0][0], self.hidden[0][1])
			hidden.append(random.sample(sample_list, sample_size))
			#print(hidden)
			self.num_active_h_neurons.append(len(hidden[i]))
			self.num_pos_h_neurons.append(self.hidden[0][1])
			for j in range(0,self.hidden[0][1]):
				self.neuron_list.append(None)
		self.raw_hidden = self.hidden
		self.hidden = hidden
		#print(self.hidden, self.num_active_h_neurons, self.num_pos_h_neurons)
		#determine number of pos and active output neurons. Then update neuron_list
		self.outputs, self.num_pos_o_neurons, self.num_active_o_neurons = self.io_extrapolator(self.outputs)
		#print(self.outputs, self.num_active_o_neurons, self.num_pos_o_neurons)
		#print(len(self.neuron_list))

		self.connect_nerons()

	def io_extrapolator(self, io_list):
		position = 0
		tot_count = 0
		used_count = 0
		for i in range(0,len(io_list)):
			specific_used_count = 0
			ranges = io_list[i][2]
			duplicate = io_list[i][1]
			used = []
			if ranges == []:
				ranges = [0,1]
				used_count += duplicate
				specific_used_count += duplicate
			else:
				sample_list = list(range(0, ranges[1]))
				sample_size = random.randint(ranges[0], ranges[1])
				raw_used = random.sample(sample_list, sample_size)
				for item in raw_used:
					val = item*duplicate
					for j in range(0,duplicate):
						used.append(val + j)
						used_count += 1
						specific_used_count += 1
			io_list[i].append(used)
			tot_count += ranges[1]*duplicate
			specific_tot_count = ranges[1]*duplicate
			io_list[i].append([specific_used_count,specific_tot_count])
		for i in range(0,tot_count):
			self.neuron_list.append(None)
		return io_list, tot_count, used_count

	def connect_nerons(self):
		#set neurons in correct positions for inputs
		iterator = 0
		cnt = 0
		self.available.append([])
		for item in self.inputs:
			if item[3] == []:
				for i in range(0,item[1]):
					self.neuron_list[iterator] = Neuron()
					self.available[cnt].append(iterator)
					iterator += 1
			else:
				for num in item[3]:
					self.neuron_list[iterator + num] = Neuron()
					self.available[cnt].append(iterator + num)
				iterator += item[4][1]

		#set neurons in correct positions for hidden
		layer = 0
		#print(self.hidden)
		for layer_list in self.hidden:
			cnt += 1
			self.available.append([])
			for num in layer_list:
				self.neuron_list[iterator+num] = Neuron()
				self.available[cnt].append(iterator + num)
			iterator += self.num_pos_h_neurons[layer]
			layer += 1

		#set neurons in correct positions for outputs
		cnt += 1
		c = 0
		self.available.append([])
		for item in self.outputs:
			if item[3] == []:
				for i in range(0,item[1]):
					self.neuron_list[iterator] = Neuron()
					self.neuron_list[iterator].label = self.outputs[c][0]
					self.available[cnt].append(iterator)
					iterator += 1
			else:
				for num in item[3]:
					self.neuron_list[iterator + num] = Neuron()
					self.neuron_list[iterator + num].label = self.outputs[c][0]
					self.available[cnt].append(iterator + num)
				iterator += item[4][1]
			c += 1

		#print(self.available, len(self.available[0]), len(self.available[1]), len(self.available[2]), len(self.available[3]))
		#iterate through neuron_list and start generating neurons connections
		#connect between hidden layer 1 and input
		neuron_range = self.available[0]
		range_size = len(self.available[0])
		for i in range(self.num_pos_i_neurons,self.num_pos_i_neurons+self.num_pos_h_neurons[0]):
			if self.neuron_list[i] != None:
				inbounds = random.sample(neuron_range, random.randint(0,range_size))
				self.num_connections += len(inbounds)
				for num in inbounds:
					neuron_pos = num
					neuron_weight = ((random.random()*2.0)-1.0)
					self.neuron_list[i].inbound.append([num,neuron_weight])

		#if interconnected, connect between hidden layer 1 and itself
		layer = 0
		if self.raw_hidden[1] == 0:
			neuron_range = self.available[1]
			range_size = len(self.available[1])
			for i in range(self.num_pos_i_neurons,self.num_pos_i_neurons+self.num_pos_h_neurons[0]):
				if self.neuron_list[i] != None:
					inbounds = random.sample(neuron_range, random.randint(0,range_size))
					self.num_connections += len(inbounds)
					for num in inbounds:
						neuron_pos = num
						neuron_weight = ((random.random()*2.0)-1.0)
						self.neuron_list[i].inbound.append([num,neuron_weight])
		#if not interconnected, connect between preceding hidden layers
		else:
			for layer in range(1,self.raw_hidden[1]):
				neuron_range = self.available[layer]
				range_size = len(self.available[layer])
				for j in range(self.num_pos_i_neurons+self.num_pos_h_neurons[layer]*(layer),self.num_pos_i_neurons+self.num_pos_h_neurons[layer]*(layer+1)):
					if self.neuron_list[j] != None:
						inbounds = random.sample(neuron_range, random.randint(0,range_size))
						self.num_connections += len(inbounds)
						for num in inbounds:
							neuron_pos = num
							neuron_weight = ((random.random()*2.0)-1.0)
							self.neuron_list[j].inbound.append([num,neuron_weight])

		#connect between last hidden layer and output
		neuron_range = self.available[layer+1]
		range_size = len(self.available[layer+1])
		for i in range(self.num_pos_i_neurons + sum(self.num_pos_h_neurons),self.num_pos_i_neurons + sum(self.num_pos_h_neurons) + self.num_pos_o_neurons):
			if self.neuron_list[i] != None:
				inbounds = random.sample(neuron_range, random.randint(0,range_size))
				self.num_connections += len(inbounds)
				for num in inbounds:
					neuron_pos = num
					neuron_weight = ((random.random()*2.0)-1.0)
					self.neuron_list[i].inbound.append([num,neuron_weight])

	def show(self):
		MULT = 3
		SIZE = WIDTH, HEIGHT = 320*MULT, 240*MULT
		black = 0, 0, 0
		white = 255,255,255

		screen = pygame.display.set_mode(SIZE)

		screen.fill(black)
		#screen.blit(ball, ballrect)
		#pre initiatializing the IO neuron variables
		neuron_pos = []
		io_radius = 3*MULT
		i_sep = (HEIGHT - io_radius*2*self.num_active_i_neurons)/(self.num_active_i_neurons+2) + io_radius*2
		o_sep = (HEIGHT - io_radius*2*self.num_active_o_neurons)/(self.num_active_o_neurons+2) + io_radius*2

		iterator = 0
		place = 0
		#Drawing all the I neurons
		for x in range(0,self.num_pos_i_neurons):
			if self.neuron_list[iterator + x] != None:
				pos = [int(i_sep),int(i_sep + (place+.5)*i_sep)]
				place += 1
				self.neuron_list[iterator + x].pos = pos
				color = [255,255,255]
				color = [int(a * self.neuron_list[iterator + x].cur_value) for a in color]
				#neuron_pos.append(pos)
				pygame.draw.circle(screen, color, pos, io_radius, 0)
		iterator += self.num_pos_i_neurons
		#print(iterator, self.num_pos_i_neurons)
		#If interconnected, drawing all the interconnected hidden neurons
		if self.raw_hidden[1] == 0:
			num_hidden_neurons = self.num_active_h_neurons[0]
			center = int(WIDTH/2), int(HEIGHT/2)
			path_radius = center[1]-int(i_sep)
			radius = int(3*MULT)
			angle = 0.0
			#place = 0
			angle_spread = 2*math.pi/num_hidden_neurons
			for j in range(0,self.num_pos_h_neurons[0]):
				if self.neuron_list[iterator + j] != None:
					x = int(math.cos(angle)*path_radius)
					y = int(math.sin(angle)*path_radius)
					pos = [center[0]+x, center[1]+y]
					#place += 1
					self.neuron_list[iterator + j].pos = pos
					color = [255,255,255]
					color = [int(a * self.neuron_list[iterator + j].cur_value) for a in color]
					#neuron_pos.append(pos)
					pygame.draw.circle(screen, color, pos, radius, 0)
					angle += angle_spread
					j += 1
			iterator += self.num_pos_h_neurons[0]
		else:
			radius = int(3*MULT)
			center = int(WIDTH/2), int(HEIGHT/2)
			dist_x_mult = .6
			dist_y_mult = .8
			x = WIDTH*dist_x_mult/2
			x_offset = WIDTH*dist_x_mult/self.raw_hidden[1]
			x = center[0] - x + x_offset/2
			for j in range(0,self.raw_hidden[1]):
				y = HEIGHT*dist_y_mult/2
				y_offset = HEIGHT*dist_y_mult/len(self.available[j+1])
				y = center[1] - y + y_offset/2
				#place = 0
				for i in range(0,self.num_pos_h_neurons[j]):
					if self.neuron_list[iterator + i] != None:
						pos = [int(x),int(y)]
						#place += 1
						self.neuron_list[iterator + i].pos = pos
						color = [255,255,255]
						color = [int(a * self.neuron_list[iterator + i].cur_value) for a in color]
						#neuron_pos.append(pos)
						pygame.draw.circle(screen, color, pos, radius, 0)
						y += y_offset
				iterator += self.num_pos_h_neurons[j]
				x += x_offset

		#Drawing all the O neurons
		place = 0
		#print(self.num_pos_o_neurons)
		for x in range(0,self.num_pos_o_neurons):
			if self.neuron_list[iterator + x] != None:
				#print(iterator, x, self.neuron_list[iterator + x])
				pos = [WIDTH - int(i_sep),int(o_sep + (place+.5)*o_sep)]
				place += 1
				self.neuron_list[iterator + x].pos = pos
				#neuron_pos.append(pos)
				color = [255,255,255]
				color = [int(a * self.neuron_list[iterator + x].cur_value) for a in color]
				pygame.draw.circle(screen, color, pos, io_radius, 0)
		iterator += self.num_pos_o_neurons
		if iterator != len(self.neuron_list):
			print("FUCK_________________FUCK")
		
		#Drawing all the connections between the neurons
		for i in range(self.num_pos_i_neurons, len(self.neuron_list)):
			if self.neuron_list[i] != None:
				coord_s = self.neuron_list[i].pos
				for j in range(0,len(self.neuron_list[i].inbound)):
					pos = self.neuron_list[i].inbound[j][0]
					weight = self.neuron_list[i].inbound[j][1]
					coord_f = self.neuron_list[pos].pos
					if weight > 0:
						pygame.draw.line(screen, [0,int(255*weight),0], coord_s, coord_f, 1)
					elif weight < 0:
						pygame.draw.line(screen, [int(-1*255*weight),0,0], coord_s, coord_f, 1)
		
		pygame.display.flip()
		
		
		flag = 0
		while 1:
			for event in pygame.event.get():
				if event.type == pygame.QUIT: 
					flag = 1
			if flag == 1:
				break
		'''
		
		while 1:
			for event in pygame.event.get():
				if event.type == pygame.QUIT: 
					sys.quit()
		'''
		return	


	def think(self, think_cycles):
		#set input neurons that are based off output neurons using label set in main
		#print(self.inputs)
		#print(self.outputs)
		i_iterator = 0
		#print(len(self.inputs))
		for i in range(0,len(self.inputs)):
			#print('_____LOOP_____')
			o_iterator = self.num_pos_i_neurons+sum(self.num_pos_h_neurons)
			#print ("out", o_iterator)
			for o in range(0,len(self.outputs)):
				if self.inputs[i][0] == self.outputs[o][0]:
					#print("match between input pos " + str(i_iterator) + " and output pos " + str(o_iterator))
					self.neuron_list[i_iterator].cur_value = self.neuron_list[o_iterator].cur_value
				o_iterator += self.outputs[o][4][1]
			#print ("in", i_iterator)
			i_iterator += self.inputs[i][4][1]

		#Neuron values will update for think_cycle number of cycles
		#think_cycles is important because if it is an interconnected brain, some neurons
		#may have an input that is affected by their output. So adding cycles helps values converge
		for i in range(0,think_cycles):
			#print("____________________")
			for j in range(self.num_pos_i_neurons,self.num_pos_i_neurons+sum(self.num_pos_h_neurons)+self.num_pos_o_neurons):
				#print(j)
				if self.neuron_list[j] != None:
					self.neuron_list[j].raw_next_value = 0.0
					for k in range(0,len(self.neuron_list[j].inbound)):
						pos = self.neuron_list[j].inbound[k][0]
						weight = self.neuron_list[j].inbound[k][1]
						val = self.neuron_list[pos].cur_value
						self.neuron_list[j].raw_next_value += val*weight
						#print(pos, weight, val, self.dna.neuron_array[j].raw_next_value)
					#if j in range(self.dna.num_neurons-3,self.dna.num_neurons):
						#print(self.dna.neuron_array[j].raw_next_value)
					#print(self.dna.neuron_array[j].raw_next_value)
					self.neuron_list[j].cur_value = sigmoid(self.neuron_list[j].raw_next_value) #if you dont want to use a sigmoid function, change here
		return

	def mutate(self, variability, weight_change_prob, connection_change_prob, neuron_change_prob):
		#check neuron_prob, if it hits add or remove hidden neurons and their connections
		for i in range(self.num_pos_i_neurons,len(self.neuron_list)-self.num_pos_o_neurons):
			#roll the dice, check if the probability hit
			if neuron_change_prob >= random.random():
				#figure out if you wanna remove or add a neuron
				new_or_rem = random.random() - .5
				#find what layer the neuron is in
				layer = int((i-self.num_pos_i_neurons)/self.num_pos_h_neurons[0])+1
				for y in range(1, len(self.available)-1):
					if i in self.available[y]:
						layer = y
						break
				neuron_min_max = self.raw_hidden[0]
				#add a neuron if you arent at max and there is no neuron there
				if new_or_rem > 0 and neuron_min_max[1] >= (self.num_active_h_neurons[layer-1] + 1) and self.neuron_list[i] == None:
					self.neuron_list[i] = Neuron()
					self.available[layer].append(i)
					self.num_active_h_neurons[layer-1] += 1
				#remove a neuron if you arent at min and the is a neuron there
				elif new_or_rem < 0 and neuron_min_max[0] <= (self.num_active_h_neurons[layer-1] - 1) and self.neuron_list[i] != None:
					self.num_active_h_neurons[layer-1] -= 1
					for j in range(0,len(self.neuron_list)):
						rem_list = []
						if self.neuron_list[j] != None:
							for k in range(0,len(self.neuron_list[j].inbound)):
								if self.neuron_list[j].inbound[k][0] == i:
									#if the neuron being deleted is an inbounds to another neuron, delete the connection
									rem_list.append(self.neuron_list[j].inbound[k])
							for item in rem_list:
								self.neuron_list[j].inbound.remove(item)
					self.available[layer].remove(i)
					self.neuron_list[i] = None

		#cycle through all connection weights, check prob of alteration and act accordingly
		for i in range(0,len(self.neuron_list)):
			if self.neuron_list[i] != None:
				for k in range(0,len(self.neuron_list[i].inbound)):
					#check if weight_change_prob hits
					if weight_change_prob >= random.random():
						#change the weight
						self.neuron_list[i].inbound[k][1] += (random.random()-.5)*variability
						if self.neuron_list[i].inbound[k][1] > 1:
							self.neuron_list[i].inbound[k][1] = 1
						elif self.neuron_list[i].inbound[k][1] < -1:
							self.neuron_list[i].inbound[k][1] = -1

		#cycle through every connection possible and check prob of creation/deletion
		#create a list of connections and whether to earase or create them. 
		#Elements will look like [[53, 24], 0] means delete connection between neuron 53 and neuron 24
			# where 53 is the neuron that feeds its cur_value into 24
		connection_list = []
		#print(connection_list)
		#first generate connection between input and first hidden layer
		pos_from = (0,self.num_pos_i_neurons)
		pos_to = (self.num_pos_i_neurons, self.num_pos_i_neurons+self.num_pos_h_neurons[0])
		connection_list += self.gen_connection_list(connection_change_prob, pos_from, pos_to)
		#now generate connections between hidden layers
		#print(connection_list)
		#check if interconnected or not, act accordingly
		if self.raw_hidden[1] == 0:
			#interconnected
			pos_from = (self.num_pos_i_neurons,self.num_pos_i_neurons+self.num_pos_h_neurons[0])
			pos_to = (self.num_pos_i_neurons, self.num_pos_i_neurons+self.num_pos_h_neurons[0])
			connection_list += self.gen_connection_list(connection_change_prob, pos_from, pos_to)
		else:
			#normal layers
			for i in range(0, self.raw_hidden[1]-1):
				pos_from = (self.num_pos_i_neurons+self.num_pos_h_neurons[0]*i,self.num_pos_i_neurons+self.num_pos_h_neurons[0]*(i+1))
				pos_to = (self.num_pos_i_neurons+self.num_pos_h_neurons[0]*(i+1), self.num_pos_i_neurons+self.num_pos_h_neurons[0]*(i+2))
				connection_list += self.gen_connection_list(connection_change_prob, pos_from, pos_to)
		#print(connection_list)
		#lastly generate connections between last hidden layer and output
		pos_from = (len(self.neuron_list)-self.num_pos_o_neurons-self.num_pos_h_neurons[0],len(self.neuron_list)-self.num_pos_o_neurons)
		pos_to = (len(self.neuron_list)-self.num_pos_o_neurons, len(self.neuron_list))
		connection_list += self.gen_connection_list(connection_change_prob, pos_from, pos_to)
		#print(connection_list)

		#iterate through connection_list and remove/add connections as it specifies
		for item in connection_list:
			s_neuron = item[0][0]
			e_neuron = item[0][1]
			rem_or_add = item[1]
			if self.neuron_list[s_neuron] != None and self.neuron_list[e_neuron] != None:
				#print(rem_or_add,s_neuron)
				#print(self.neuron_list[e_neuron].inbound)
				if rem_or_add:
					#add if not already there
					if s_neuron not in [item[0] for item in self.neuron_list[e_neuron].inbound]:
						self.neuron_list[e_neuron].inbound.append([s_neuron, random.random()-.5])
						self.num_connections += 1
				else:
					#remove if it exists
					if s_neuron in [item[0] for item in self.neuron_list[e_neuron].inbound]:
						pos = [item[0] for item in self.neuron_list[e_neuron].inbound].index(s_neuron)
						del self.neuron_list[e_neuron].inbound[pos]
						self.num_connections -= 1
				#print(self.neuron_list[e_neuron].inbound)
				#exit()
		return

	def change_input(self, label, add_or_rem):
		offset = 0
		size = 0
		dup = 0
		smallest = 0
		if self.num_active_i_neurons + add_or_rem <= self.num_pos_i_neurons and self.num_active_i_neurons + add_or_rem >= 1:
			for i in range(0,len(self.inputs)):
				if label == self.inputs[i][0]:
					#print("changing " + label, add_or_rem, self.inputs)
					size = int(self.inputs[i][4][1])
					dup = int(self.inputs[i][1])
					smallest = int(int(self.inputs[i][4][0])/dup)
					break
				else:
					offset += self.inputs[i][4][1]
			#print(dup)
			#check every position that is 'dup' apart in neuron_list
			#create list of empty neurons if add, and a list of full neurons if rem
			pick_list = []
			#print("offset", offset)
			#print(self.neuron_list[1:offset+size-1])
			i = offset
			while i < offset+size:
				#print(i)
				#x = (i-offset)
				if add_or_rem > 0:
					if self.neuron_list[i] == None:
						pick_list.append(i)
				else: 
					if self.neuron_list[i] != None:
						pick_list.append(i)
				i += dup

			#print(pick_list)
			num = len(pick_list)
			if add_or_rem > 0:
				if add_or_rem > num:
					add_or_rem = num
				choice = random.sample(pick_list,add_or_rem)
				for item in choice:
					for i in range(0,dup):
						self.neuron_list[item+i] = Neuron()
						self.available[0].append(item+i)
					self.num_active_i_neurons += dup
			else:
				#print(self.inputs)
				add_or_rem = -1*add_or_rem
				if add_or_rem > num-smallest:
					add_or_rem = num-smallest
				#print(num, add_or_rem, smallest)
				#print("_____________")
				raw_choice = random.sample(pick_list,add_or_rem)
				#print(raw_choice)
				choice = []
				for item in raw_choice:
					for i in range(0,dup):
						choice.append(item+i)
				#print(choice)
				for item in choice:
					for i in range(0,len(self.neuron_list)):
						if self.neuron_list[i] != None:
							#print(self.neuron_list[i].inbound)
							for x in range(0, len(self.neuron_list[i].inbound)):
								if item == self.neuron_list[i].inbound[x][0]:
									del self.neuron_list[i].inbound[x]
									self.num_connections -= 1
										#print("removed")
									break
							#print(self.neuron_list[i].inbound)
					self.neuron_list[item] = None
					self.available[0].remove(item)
					self.num_active_i_neurons -= 1
			#randomly pick the how many from the list you want to remove
			#if add, add to those neurons. if rem, remove those neurons and connections
			##
			#3



		return

	def gen_connection_list(self, connection_change_prob, pos_from, pos_to):
		connect_list = []
		for i in range(pos_from[0],pos_from[1]):
			for k in range(pos_to[0],pos_to[1]):
				if connection_change_prob >= random.random():
					rem_or_add = random.random()-.5 
					if rem_or_add > 0:
						rem_or_add = 1
					else:
						rem_or_add = 0
					connect_list.append([(i,k),rem_or_add])
		return connect_list


	def get_outputs(self):
		#return a list of [label, cur_value] pairs for all outputs
		outputs = []
		#print(self.num_pos_o_neurons, self.num_active_o_neurons)
		#print(len(self.neuron_list)-self.num_pos_o_neurons, len(self.neuron_list))
		#print(self.neuron_list[len(self.neuron_list)-self.num_pos_o_neurons])
		#print("Starts at", len(self.neuron_list)-self.num_pos_o_neurons, len(self.neuron_list))
		print(self.num_pos_o_neurons, self.num_active_o_neurons)
		print(self.num_pos_i_neurons, self.num_active_i_neurons)
		for i in range(len(self.neuron_list)-self.num_pos_o_neurons, len(self.neuron_list)):
			if self.neuron_list[i] != None:
				outputs.append([self.neuron_list[i].label,self.neuron_list[i].cur_value])
			else:
				outputs.append(None)
		return outputs

	def set_inputs(self, inputs_list):
		x = 0
		if self.num_active_i_neurons != len(inputs_list):
			print(self.num_active_i_neurons,len(inputs_list))
			print("List size not same size as avialable NN inputs")
			exit()
		for i in range(0, self.num_pos_i_neurons):
			if self.neuron_list[i] != None:
				self.neuron_list[i].cur_value = inputs_list[x]
				x += 1
		return 



