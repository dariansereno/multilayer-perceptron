import numpy as np
#https://cs231n.github.io/neural-networks-case-study/
def create_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4, samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    split_index = int(len(X) *  0.8)

    # Split the data into training and testing sets
    train_X = X[:split_index]
    train_Y = y[:split_index]
    test_X = X[split_index:]
    test_Y = y[split_index:]
    #test_Y = test_Y.reshape(-1, 1)
    #train_Y = train_Y.reshape(-1, 1)
    return (train_X, train_Y), (test_X, test_Y)

create_data(1000, 5)