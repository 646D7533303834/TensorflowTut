import tensorflow as tf
import numpy as np
import random

def main():
	
	class_label_index = {
		'A':0,
		'B':1,
		'C':2,
		'D':3,
		'E':4,
		'F':5,
		'H':6
	}
	
	spot_size_label_index = {
		'X':0,
		'R':1,
		'S':2,
		'A':3,
		'H':4,
		'K':5
	}
	
	spot_dist_label_index = {
		'X':0,
		'O':1,
		'I':2,
		'C':3
	}
	
	data = []
	labels = []
	
	with open("flare.data2", "r") as input_file:
		for line in input_file:
			if(len(line.strip()) == 0):
				continue
				
			full_data_line = line.strip().split(' ')
			
			data_line = full_data_line[0:-3]
			predicted_line = full_data_line[-3:]
			
			x = data_line[0]
			
			if x in class_label_index:
				data_line[0] = class_label_index[x]
			else:
				print("Bad data", line)
				continue
			
			x = data_line[1]			
			if x in spot_size_label_index:
				data_line[1] = spot_size_label_index[x]
			else:
				print("Bad data", line)
				continue
				
			x = data_line[2]
			if x in spot_dist_label_index:
				data_line[2] = spot_dist_label_index[x]
			else:
				print("Bad data", line)
				continue
			
			data_line = list(map(float, data_line))
			##data_line = tf.nn.l2_normalize(data_line, 2, 0)
			predicted_line = list(map(int, predicted_line))
			
			data.append(data_line)
			labels.append(predicted_line)
			
	print("data", len(data))
	print("labels", len(labels))
	print(data[0])
	print(labels[0])
	
	dataset = list(zip(data, labels))
	random.shuffle(dataset)
	test_length = int(len(dataset) * 0.67)
	
	print("test_length", test_length)
	train_dataset = dataset[:test_length]
	test_dataset = dataset[test_length:]
	
	x_size = 10
	y_size = 3
	
	#Symbols
	inputs = tf.placeholder("float", shape=[None, x_size])
	labels = tf.placeholder("float", shape=[None, y_size])
	
	weights1 = tf.get_variable("weight1", shape=[x_size, 500], initializer=tf.contrib.layers.xavier_initializer())
	bias1 = tf.get_variable("bias1", shape=[500], initializer=tf.constant_initializer(value=0.0))
	
	layer1 = tf.nn.relu(tf.matmul(inputs, weights1) + bias1)
	
	weights2 = tf.get_variable("weight2", shape=[500, 500], initializer=tf.contrib.layers.xavier_initializer())
	bias2 = tf.get_variable("bias2", shape=[500], initializer=tf.constant_initializer(value=0.0))

	layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + bias2)
	
	weights3 = tf.get_variable("weight3", shape=[500, 500], initializer=tf.contrib.layers.xavier_initializer())
	bias3 = tf.get_variable("bias3", shape=[500], initializer=tf.constant_initializer(value=0.0))
	
	layer3 = tf.nn.relu(tf.matmul(layer2, weights3) + bias3)
	
	weights4 = tf.get_variable("weight4", shape=[500, 500], initializer=tf.contrib.layers.xavier_initializer())
	bias4 = tf.get_variable("bias4", shape=[500], initializer=tf.constant_initializer(value=0.0))
	
	layer4 = tf.nn.relu(tf.matmul(layer3, weights4) + bias4)
	
	weights5 = tf.get_variable("weight5", shape=[500, 250], initializer=tf.contrib.layers.xavier_initializer())
	bias5 = tf.get_variable("bias5", shape=[250], initializer=tf.constant_initializer(value=0.0))
	
	layer5 = tf.nn.relu(tf.matmul(layer4, weights5) + bias5)
	
	weights6 = tf.get_variable("weight6", shape=[250, 100], initializer=tf.contrib.layers.xavier_initializer())
	bias6 = tf.get_variable("bias6", shape=[100], initializer=tf.constant_initializer(value=0.0))

	layer6 = tf.nn.relu(tf.matmul(layer5, weights6) + bias6)
	
	weights7 = tf.get_variable("weight7", shape=[100, 3], initializer=tf.contrib.layers.xavier_initializer())
	bias7 = tf.get_variable("bias7", shape=[3], initializer=tf.constant_initializer(value=0.0))
	
	##outputs = tf.round(tf.matmul(layer5, weights6) + bias6)
	outputs = tf.matmul(layer6, weights7) + bias7
	outputs_rounded = tf.round(tf.matmul(layer6, weights7) + bias7)
	##print("outputs type", type(outputs))
	
	#backprop
	##loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=outputs))
	loss = tf.reduce_mean(tf.square(labels - outputs))
	train = tf.train.AdamOptimizer().minimize(loss)
	
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		
		avg_accuracy = 0.0;
		
		for epoch in range(20000):
			##batch = train_dataset[:5] 
			batch = random.sample(train_dataset, 100)
			input_batch, labels_batch = zip(*batch)
			loss_output, prediction_output, _ = sess.run([loss, outputs_rounded, train], feed_dict={inputs: input_batch, labels: labels_batch})
			
			#if sum(labels_batch[0]) > 3:
			#	print("prediction_output", prediction_output[0])
			#	print("labels_batch", labels_batch[0])
			#	print("error", (labels_batch - prediction_output)[0])
			#
			#	##accuracy = np.divide(prediction_output, labels_batch)
			#	accuracy = np.mean(labels_batch == prediction_output)
			#
			#	print("train", "loss", loss_output, "accuracy", accuracy)
			accuracy = np.mean(labels_batch == prediction_output)
			avg_accuracy = ((avg_accuracy * epoch) + accuracy)/(epoch + 1)
			print("train", "loss", loss_output)
			print("accuracy", accuracy)
			print("average accuracy", avg_accuracy)
	
if __name__ == "__main__":
	main()
			