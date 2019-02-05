import numpy as np
import matplotlib.pyplot as plt
data_file=open('mnist_train_100.csv','r')
data_list=data_file.readlines()
all_values = data_list[0].split(',')
image_array=np.asfarray(all_values[1:]).reshape(28,28)
plt.imshow(image_array,cmap='Greys',interpolation='None')
plt.show()
data_file.close()
