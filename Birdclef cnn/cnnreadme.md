For the model implemented with birdclef
We chose a 1d-CNN (convulutional neural network). It would just take the existing data from the new_train.csv file and remove the labels that were not needed
such as filename, rating, url and others that would not help the model learn. It was then encoded to numeric form for processing using label encoder.
Then the numeric labels were converted into one-hot encoded format for the multi-class classification and the data was scaled using StandardScaler that normalizes the feature data whereby each feature would have 0 mean and variance which should improve model performance.
New_X was done to give a new shape to the scaled values which was needed to use Conv1d as it requires 3 dimension
ReLU was a function used to allow the model to learn more complex patterns and softmax would output the probability distribution.
The code uses Adam optimizer which is stohastic gradient descent function to adjust learning rates. 
Additionally Categorial cross entropy is the loss function to compare predicted p robabilities with one-hot encoded labels.
Therefore the model learns by training on the data and adjusting the weights using both Adam and backpropagation.
The features are extracted from the input data using the CNN and then that i s used by the dense layer to classify the input into one of the target classes. Finally it minimizes the categorical cross entropy loss to improve the accuracy.