import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle




def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    

def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sig = 1/(1 + np.exp(-z))
    return  sig
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label = np.array([])
    
    print "\nStarting preprocess function:"

    #Training Data Stacking
    tr0 = mat.get('train0')
    tr1 = mat.get('train1')
    tr2 = mat.get('train2')
    tr3 = mat.get('train3')
    tr4 = mat.get('train4')
    tr5 = mat.get('train5')
    tr6 = mat.get('train6')
    tr7 = mat.get('train7')
    tr8 = mat.get('train8')
    tr9 = mat.get('train9')
    train_comb = np.concatenate([tr0,tr1,tr2,tr3,tr4,tr5,tr6,tr7,tr8,tr9])
    train_data = train_comb
    #train_data = np.asmatrix(train_comb)
    #print train_data
    #traindatasize = train_data.shape
    #print traindatasize
    
    #Test Data Stacking
    tt0 = mat.get('test0')
    tt1 = mat.get('test1')
    tt2 = mat.get('test2')
    tt3 = mat.get('test3')
    tt4 = mat.get('test4')
    tt5 = mat.get('test5')
    tt6 = mat.get('test6')
    tt7 = mat.get('test7')
    tt8 = mat.get('test8')
    tt9 = mat.get('test9')
    test_comb = np.concatenate([tt0,tt1,tt2,tt3,tt4,tt5,tt6,tt7,tt8,tt9])  
    test_data = test_comb
    #test_data = np.asmatrix(test_comb)
    #print test_data
    #testdatasize = test_data.shape
    #print testdatasize
    
    #Training Data Labels Stacking
    tr0_size = tr0.shape[0]
    trlb0 = np.full((tr0_size,1),0)
    tr1_size = tr1.shape[0]
    trlb1 = np.full((tr1_size,1),1)
    tr2_size = tr2.shape[0]
    trlb2 = np.full((tr2_size,1),2)
    tr3_size = tr3.shape[0]
    trlb3 = np.full((tr3_size,1),3)
    tr4_size = tr4.shape[0]
    trlb4 = np.full((tr4_size,1),4)
    tr5_size = tr5.shape[0]
    trlb5 = np.full((tr5_size,1),5)
    tr6_size = tr6.shape[0]
    trlb6 = np.full((tr6_size,1),6)
    tr7_size = tr7.shape[0]
    trlb7 = np.full((tr7_size,1),7)
    tr8_size = tr8.shape[0]
    trlb8 = np.full((tr8_size,1),8)
    tr9_size = tr9.shape[0]
    trlb9 = np.full((tr9_size,1),9)
    trlb_comb = np.concatenate([trlb0,trlb1,trlb2,trlb3,trlb4,trlb5,trlb6,trlb7,trlb8,trlb9])
    train_label = np.asmatrix(trlb_comb)
    #print train_label
    #trainlabelsize = train_label.shape
    #print trainlabelsize
    
    #Test Data Labels Stacking
    tt0_size = tt0.shape[0]
    ttlb0 = np.full((tt0_size,1),0)
    tt1_size = tt1.shape[0]
    ttlb1 = np.full((tt1_size,1),1)
    tt2_size = tt2.shape[0]
    ttlb2 = np.full((tt2_size,1),2)
    tt3_size = tt3.shape[0]
    ttlb3 = np.full((tt3_size,1),3)
    tt4_size = tt4.shape[0]
    ttlb4 = np.full((tt4_size,1),4)
    tt5_size = tt5.shape[0]
    ttlb5 = np.full((tt5_size,1),5)
    tt6_size = tt6.shape[0]
    ttlb6 = np.full((tt6_size,1),6)
    tt7_size = tt7.shape[0]
    ttlb7 = np.full((tt7_size,1),7)
    tt8_size = tt8.shape[0]
    ttlb8 = np.full((tt8_size,1),8)
    tt9_size = tt9.shape[0]
    ttlb9 = np.full((tt9_size,1),9)
    ttlb_comb = np.concatenate([ttlb0,ttlb1,ttlb2,ttlb3,ttlb4,ttlb5,ttlb6,ttlb7,ttlb8,ttlb9])
    test_label_ini = np.asmatrix(ttlb_comb)
    #print test_label
    #testlabelsize = test_label.shape
    #print testlabelsize
    test_label = np.empty([test_label_ini.shape[0],])
    for i in range(test_label_ini.shape[0]):
        test_label[i,] = test_label_ini[i,0]
    
    print "\tTa-da! Stacking Successful!"
    
    #Normalization
    for i in train_data.flat:
        i = float(i)
        i = i/255 
    for j in train_data.flat:
        j = float(j)
        j = j/255
    print "\tTa-da! Normalization Successful!"
    
    #Splitting    
    a = range(train_data.shape[0])
    aperm = np.random.permutation(a)
    validation_data = train_data[aperm[0:10000],:]
    train_data = train_data[aperm[10000:],:]
    validation_label_ini = train_label[aperm[0:10000],:]
    train_label_ini = train_label[aperm[10000:],:]
    validation_label = np.empty([validation_data.shape[0],])
    train_label = np.empty([train_data.shape[0],])
    for i in range(validation_label.shape[0]):
        validation_label[i,] = validation_label_ini[i,0]
    for i in range(train_label.shape[0]):
        train_label[i,] = train_label_ini[i,0]
    print "\tTa-da! Splitting Successful!"
    
    #Feature Selection
    rows_to_delete = []
    for col in range(train_data.shape[1]):
        ctr = 0
        for row in range(train_data.shape[0]):
            if (train_data[0,col] == train_data[row,col]):
                ctr+=1
        if ctr == train_data.shape[0]:
            rows_to_delete.append(col)
    train_data = np.delete(train_data, rows_to_delete, axis=1)
    print "\tTa-da! Feature Selection Successful!"
    
    print "\tInitial size of training data:" + str(train_data.shape)
    
    print "Preprocess function completed successfully!"
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    #Your code here
    
    print "\nStarting nnObjFunction:"
    
    #Forward Pass
    print "\tSize of training data before appending:" + str(training_data.shape)
    bias = np.ones((training_data.shape[0],1))
    training_data_appended = np.append(training_data, bias, 1)
    print "\tSize of training data after appending:" + str(training_data_appended.shape)
    ip_to_hidden = np.dot(training_data_appended,w1.T)
    print "\tDot product successful!"
    print "\tSize of dot product after input layer:" + str(ip_to_hidden.shape)
    sig_hid = sigmoid(ip_to_hidden)
    print "\tSigmoid successful!"
    print "\tSize of sigmoid output at hidden layer:" + str(sig_hid.shape)
    bias_h = np.ones((sig_hid.shape[0],1))
    sig_hid_appended = np.append(sig_hid, bias_h, 1)
    print "\tSize of sigmoid output at hidden layer after appending:" + str(sig_hid_appended.shape)
    hidden_to_op = np.dot(sig_hid_appended,w2.T)
    sig_op = sigmoid(hidden_to_op)
    print "\tSize of sigmoid output at output layer:" + str(sig_op.shape)

    #One-to-K encoding
    one_to_k = np.zeros([50000,10])
    for j in range(training_label.shape[0]):
        if (training_label[j,] == 0):
            one_to_k[j,0] = 1
        elif (training_label[j,] == 1):
            one_to_k[j,1] = 1
        elif (training_label[j,] == 2):
            one_to_k[j,2] = 1
        elif (training_label[j,] == 3):
            one_to_k[j,3] = 1
        elif (training_label[j,] == 4):
            one_to_k[j,4] = 1
        elif (training_label[j,] == 5):
            one_to_k[j,5] = 1
        elif (training_label[j,] == 6):
            one_to_k[j,6] = 1
        elif (training_label[j,] == 7):
            one_to_k[j,7] = 1
        elif (training_label[j,] == 8):
            one_to_k[j,8] = 1
        elif (training_label[j,] == 9):
            one_to_k[j,9] = 1
#    np.savetxt('onetok.txt', one_to_k)
    
    #Error Value
    #Equation 6
    motha_sum = 0
    temp1 = one_to_k - sig_op    
    for row in range(50000):
        chhota_sum = 0
        for col in range(10):
            chhota_sum = chhota_sum + ((temp1[row,col])**2)
        chhota_sum = (chhota_sum/2)
        motha_sum = motha_sum + chhota_sum
    eq6 = motha_sum/50000
    print "\tTotal error for the entire training data:" + str(eq6)
    
    #Equation 15
    sum_sq_w1 = 0
    sum_sq_w2 = 0
    for row in range(w1.shape[0]):
        for col in range(w1.shape[1]):
            sum_sq_w1 = sum_sq_w1 + ((w1[row,col])**2)
    for row in range(w2.shape[0]):
        for col in range(w2.shape[1]):
            sum_sq_w2 = sum_sq_w2 + ((w2[row,col])**2)
    eq15 = eq6 + np.dot((lambdaval/(2*training_data.shape[0])),(sum_sq_w1+sum_sq_w2)) 
    #eq15 = eq6 + ((lambdaval/(2*training_data.shape[0]))*(sum_sq_w1+sum_sq_w2))
    print "eq15:"
    print eq15
    obj_val = eq15
    
    #Error Grad
    #Equation 8
    all_ones_eq8 = np.ones([training_data.shape[0],n_class])
    temp1 = one_to_k - sig_op
    temp2 = all_ones_eq8 - sig_op
    temp = np.multiply(temp1,temp2)
    dl = np.multiply(temp,sig_op)
#    dl = np.zeros([50000,10])
#    for row in range(50000):
#        for col in range(10):
#            elem = np.dot(np.dot(temp1[row,col],temp2[row,col]),sig_op[row,col])
#            dl[row,col] = elem
#    eq8a=np.dot((sig_hid_appended.T),dl)
#    eq8 = -1*eq8a
#    #eq8 = -1*(sig_hid.T)*dl
    eq8 = np.dot(sig_hid_appended.T,dl)
    eq8 = -1*eq8
    print "eq8 shape:"
    print eq8.shape
    
    #Equation 12
#    all_ones_eq12 = np.ones([50000,(n_hidden+1)])
#    eq12a=((all_ones_eq12 - sig_hid_appended).T)
#    eq12b=np.dot(eq12a,sig_hid_appended)
#    eq12p1 = (-1*eq12b)
#    #eq12p1 = -1*((all_ones_eq12 - sig_hid).T)*(sig_hid)
#    eq12p2a=(np.dot(dl,w2)).T
#    eq12p2=np.dot(eq12p2a,training_data_appended)
#    #eq12p2 = ((np.dot(dl,w2_copy)).T)*training_data
#    eq12 = np.dot(eq12p1,eq12p2)
    all_ones_eq12 = np.ones([sig_hid_appended.shape[0],(n_hidden+1)])
    eq12a = all_ones_eq12 - sig_hid_appended
    eq12b = sig_hid_appended
    eq12ab = np.multiply(eq12a,eq12b)
    eq12c = np.dot(dl,w2)
    eq12p1 = np.multiply(eq12ab,eq12c)
    eq12 = np.dot((eq12p1.T),training_data_appended)
    val = -1
    eq12 = np.multiply(eq12,val)
    print "eq12 initial shape:"
    print eq12.shape
    row_to_delete = [n_hidden]
    eq12 = np.delete(eq12, row_to_delete, axis=0)
    print "eq12 final shape:"
    print eq12.shape
    
    
    #Equation 17
    eq17a = lambdaval*w1
    eq17 = eq12 + eq17a
    eq17 = eq17.T
    eq17 = eq17/50000
    print "eq17 shape:"
    print eq17.shape
    eq17 = eq17.T
    
    #Equation 16
    eq16a = lambdaval*w2
    eq16b = eq16a.T
    eq16 = eq8 + eq16b
    eq16 = eq16/50000
    print "eq16 shape:"
    print eq16.shape
    eq16 = eq16.T
    
    grad_w2 = eq16
    grad_w1 = eq17
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    print "\ngradw1 shape:"
    print grad_w1.shape
    print "\ngradw2 shape:"
    print grad_w2.shape
    print "\nobj_grad shape"
    print obj_grad.shape

#    grad_w1 = grad_w1.T
#    grad_w2 = grad_w2.T
#    print "\nNew gradw1shape:"
#    print grad_w1.shape
#    print "\nNew gradw2shape:"
#    print grad_w2.shape
#    
#    obj_grad_ini = np.concatenate((grad_w1, grad_w2), 0)
#    print "\nobj_grad_ini shape:"
#    print obj_grad_ini.shape
#    
#    obj_grad = np.empty([obj_grad_ini.shape[0],])
#    print "\nEmpty obj_grad shape:"
#    
#    for i in range(obj_grad.shape[0]):
#        obj_grad[i,] = obj_grad_ini[i,]
#    print "\nFilled obj_grad shape:"
#    print obj_grad.shape
#    np.savetxt('temp_op.txt', obj_grad)
    
#==============================================================================
#     f = open('temp_op.txt','a')
#     f.write(one_to_k)
#     f.close
#==============================================================================
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])
    #obj_val = 0
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    #Your code here
    print "\nStarting nnPredict:"
    
    print "\tData before appending:" + str(data.shape)
    bias = np.ones((data.shape[0],1))
    data_appended = np.append(data, bias, 1)
    print "\tData after appending:" + str(data_appended.shape)
    ip_to_hidden = np.dot(data_appended,w1.T)
    print "\tDot product successful!"
    print "\tSize of dot product after input layer:" + str(ip_to_hidden.shape)
    sig_hid = sigmoid(ip_to_hidden)
    print "\tSigmoid successful!"
    print "\tSize of sigmoid output at hidden layer:" + str(sig_hid.shape)
    bias_h = np.ones((sig_hid.shape[0],1))
    sig_hid_appended = np.append(sig_hid, bias_h, 1)
    print "\tSig hid after appending:" + str(sig_hid_appended.shape)
    hidden_to_op = np.dot(sig_hid_appended,w2.T)
    sig_op = sigmoid(hidden_to_op)
    print "\tSize of sigmoid output at output layer:" + str(sig_op.shape)

    max_indices = np.argmax(sig_op, axis=1)
    print "Size of max_indices:"
    print max_indices.shape
    labels = np.empty([sig_op.shape[0],])
    for j in range(max_indices.shape[0]):
        if (max_indices[j,] == 0):
            labels[j,] = 0
        elif (max_indices[j,] == 1):
            labels[j,] = 1
        elif (max_indices[j,] == 2):
            labels[j,] = 2
        elif (max_indices[j,] == 3):
            labels[j,] = 3
        elif (max_indices[j,] == 4):
            labels[j,] = 4
        elif (max_indices[j,] == 5):
            labels[j,] = 5
        elif (max_indices[j,] == 6):
            labels[j,] = 6
        elif (max_indices[j,] == 7):
            labels[j,] = 7
        elif (max_indices[j,] == 8):
            labels[j,] = 8
        elif (max_indices[j,] == 9):
            labels[j,] = 9

    print "Size of labels:"
    print labels.shape
    print labels
#==============================================================================
#     f = open('temp_op.txt','w')
#     for i in range(labels.shape[0]):
#         f.write(labels[i,])
#     f.close
#==============================================================================
    
    return labels
    


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.2;

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\nTraining set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
#np.savetxt('training_actual.txt', train_label)
#np.savetxt('training_predicted.txt', predicted_label)


validation_data = validation_data[:,0:train_data.shape[1]]

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\nValidation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
#np.savetxt('validation_actual.txt', validation_label)
#np.savetxt('validation_predicted.txt', predicted_label)


test_data = test_data[:,0:train_data.shape[1]]
predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Test Dataset

print('\nTest set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

a = [n_hidden, w1, w2, lambdaval]
f = open('params.p','wb')
pickle.dump(a, f)
f.close()