# Answer the conceptual questions here
Q1: Is there anything we need to know to get your code to work? If you did not get your code working or handed in an incomplete solution please let us know what you did complete (0-4 sentences)

No.

Q2: Why do we normalize our pixel values between 0-1? (1-3 sentences)

In order to scale the input data so that we can increase the speed of convergence of the training process.

Q3: Why do we use a bias vector in our forward pass? (1-3 sentences)

This is to learn an offset for each neuron. This will eventually help in fitting the data better by shifting the activation function.

Q4: Why do we separate the functions for the gradient descent update from the calculation of the gradient in back propagation? (2-4 sentences)

This is to maintain modularity and clarity in code. Easy to debug, test, and reuse the algorithms. 

Q5: What are some qualities of MNIST that make it a “good” dataset for a classification problem? (2-3 sentences)

It comes with a labelled data which is a good dataset for classification problems. It is also well balanced, easy and small.

Q6: Suppose you are an administrator of the NZ Health Service (CDHB or similar). What positive and/or negative effects would result from deploying an MNIST-trained neural network to recognize numerical codes on forms that are completed by hand by a patient when arriving for a health service appointment? (2-4 sentences)

This could improve the efficiency and accuracy in data entry, reducing manual errors and save time. The negative can be that it might find it hard to recognize different variations in handwriting styles and could require additional preprocessing to handle real world data effectively.
