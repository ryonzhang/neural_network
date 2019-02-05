from neural_network import neural_network
import numpy as np
import matplotlib.pyplot as plt
input_nodes=784
hidden_nodes=100
output_nodes=10
learning_rate=0.3
n= neural_network.neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
training_data_file=open('../mnist_dataset/mnist_train_100.csv','r')
training_data_list=training_data_file.readlines()
training_data_file.close()

epoch = 1000

for _ in range(epoch):
    for record in training_data_list:
        all_values=record.split(',')
        inputs=(np.asfarray(all_values[1:])/255.0*0.99)+0.01
        targets=np.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)

test_data_file=open('../mnist_dataset/mnist_test_10.csv','r')
test_data_list=test_data_file.readlines()
test_data_file.close()

all_values = test_data_list[0].split(',')
image_array=np.asfarray(all_values[1:]).reshape(28,28)
plt.imshow(image_array,cmap='Greys',interpolation='None')
plt.show()
print(n.query(np.asfarray(all_values[1:])))

scorecard=[]
for record in test_data_list:
    all_values = record.split(',')
    correct_label=int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    if label ==correct_label :
        scorecard.append(1)
    else:
        scorecard.append(0)
print(scorecard)

