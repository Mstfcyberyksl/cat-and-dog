import numpy as np
import glob
import os
import gc
from PIL import Image
import pickle
class Layer():
    def __init__(self,input_neurons,output_neurons):
        weights,biases = np.random.rand(input_neurons,output_neurons), np.zeros((1,output_neurons))
        self.weights = weights * np.sqrt(1/input_neurons)
        self.biases = biases
        self.cache = {}
    

    def forward(self,inputs):
        z =  np.dot(inputs,self.weights) + self.biases
        a = np.maximum(0,z)
        self.cache = {'input' : inputs,'z' : z, 'a' : a}
        return a

    def update(self,weight,bias, lr):
        self.weights -= weight * lr
        self.biases -= bias * lr
def relu(input):
    return np.maximum(0,input)
def softmax(input):
    exp_vals = np.exp(input - np.max(input,axis = 1,keepdims=True))
    return exp_vals / np.sum(exp_vals,axis=1,keepdims=True)
def accuracy_calc(input,y):
    return np.mean(np.argmax(input,axis = 1) == np.argmax(y,axis = 1))
def acc_calc(input,y):
    return np.mean(input == y)
def backward(A,Y):
    return (A - Y) / A.shape[0]
def cross_entropy(A,Y):
    
    clipped = np.clip(A,1e-7,1-1e-7)
    
    return -np.mean(np.sum(np.log(clipped) * Y, axis = 1))

def compute_grads(inputs,dz):
    m = inputs.shape[0]
    dw = np.dot(inputs.T,dz) / m
    db = np.sum(dz,axis = 0,keepdims = True) 
    return dw,db
def learning_rate_change(lr,acc):
    if acc < 50:
        lr += lr * 0.1
    elif 75 < acc < 90:
        lr -= lr * 0.025
    elif 90 <= acc:
        lr  -= lr * 0.05
    return lr
def train(x_train,y_train,lr = 0.01,epochs = 2):
    global loss,accuracy
    for _ in range(epochs):
        z1 = layer1.forward(x_train)
        z2 = layer2.forward(z1)
        z3 = layer3.forward(z2)
        z4 = layer4.forward(z3)
        z5 = layer5.forward(z4)
        z6 = np.dot(z5,output.weights) + output.biases
        
        
        preds = softmax(z6)
        for i in preds:
            data = np.argmax(i)
            data = data.astype(np.int8)
            data_acc.append(data)
        for i in y_train:  
            data = np.argmax(i)
            data = data.astype(np.int8)  
            data_y.append(data)
        print("preds = " + str(preds))
        print("data_acc = " + str(data_acc))
        print("data_y = " + str(data_y))
        accuracy = acc_calc(data_acc,data_y)
        
        dz6 = backward(preds,y_train)
        dw6,db6 = compute_grads(z5,dz6)

        dz5 = np.dot(dz6,output.weights.T) * (z5 > 0)
        dw5, db5 = compute_grads(z4,dz5)

        dz4 = np.dot(dz5,layer5.weights.T) * (z4 > 0)
        dw4, db4 = compute_grads(z3,dz4)

        dz3 = np.dot(dz4,layer4.weights.T) * (z3 > 0)
        dw3, db3 = compute_grads(z2,dz3)

        dz2 = np.dot(dz3,layer3.weights.T) * (z2 > 0)
        dw2, db2 = compute_grads(z1,dz2)

        dz1 = np.dot(dz2,layer2.weights.T) * (z1 > 0)
        dw1, db1 = compute_grads(x_train,dz1)

        #learning_rate = learning_rate_change(lr,accuracy)
        learning_rate = lr
        layer1.update(dw1,db1,learning_rate)
        layer2.update(dw2,db2,learning_rate)
        layer3.update(dw3,db3,learning_rate)
        layer4.update(dw4,db4,learning_rate)
        layer5.update(dw5,db5,learning_rate)
        output.update(dw6,db6,learning_rate)
        print("accuracy = " + str(accuracy))
        if accuracy == 1:
            break


max_width = 0
max_height = 0
cat_train_folder = "/home/mustafa/Desktop/nnfs/dataset/training_set/training_set/cats"
dog_train_folder = "/home/mustafa/Desktop/nnfs/dataset/training_set/training_set/dogs"
cat_test_folder = "/home/mustafa/Desktop/nnfs/dataset/test_set/test_set/cats"
dog_test_folder = "/home/mustafa/Desktop/nnfs/dataset/test_set/test_set/dogs"

cat_train_photos = glob.glob(os.path.join(cat_train_folder,"*.jpg"))
dog_train_photos = glob.glob(os.path.join(dog_train_folder,"*.jpg"))
cat_test_photos = glob.glob(os.path.join(cat_test_folder,"*.jpg"))
dog_test_photos = glob.glob(os.path.join(dog_test_folder,"*.jpg"))

del cat_train_folder
del dog_train_folder
del cat_test_folder
del dog_test_folder
gc.collect()

cat_train = []
dog_train = []


print("Trying to find max height and width will start")
for image in cat_train_photos:
    img = Image.open(image)
    pixels = np.array(img)
    if pixels.shape[0] > max_height:
        max_height = pixels.shape[0]
    if pixels.shape[1] > max_width:
        max_width = pixels.shape[1]
for image in dog_train_photos:
    img = Image.open(image)
    pixels = np.array(img)
    if pixels.shape[0] > max_height:
        max_height = pixels.shape[0]
    if pixels.shape[1] > max_width:
        max_width = pixels.shape[1]

print("found max height and width")
print("cat_train_photos translated to numpy array will start")

zeroarray = np.zeros((max_height,max_width,3),dtype=np.float16)

for image in cat_train_photos:
    img = Image.open(image)
    img = np.array(img)
    
    padded = zeroarray.copy()
    padded[:img.shape[0], : img.shape[1], : ] = img
    padded /= 255.0
    cat_train.append(np.array(padded.flatten()))

del cat_train_photos
gc.collect()

cat_train_data = np.array(cat_train)

del cat_train
gc.collect()


print("cat_train_photos translated to numpy array ended")
print("dog_train_photos translated to numpy array will start")

for image in dog_train_photos:
    img = Image.open(image)
    img = np.array(img)
    padded = zeroarray.copy()
    padded[:img.shape[0], : img.shape[1], : ] = img
    padded /= 255.0
    dog_train.append(np.array(padded.flatten()))

del dog_train_photos
del zeroarray
del padded
gc.collect()

print("dog_train_photos translated to numpy array ended")

print("change them to numpy array will start")

dog_train_data = np.array(dog_train)


del dog_train
gc.collect()


print("change them to numpy array ended")

print("Layer initializing will start")
layer1 = Layer(max_height * max_width * 3,256)
layer2 = Layer(256,256)
layer3 = Layer(256,128)
layer4 = Layer(128,64)
layer5 = Layer(64,32)
output = Layer(32,2)
print("Layer initializing ended")

batch_size = 16
epoch = 2
loss, accuracy = 0, 0

print("training will start")
for j in range(epoch):
    data_acc = []
    data_y = []
    print("round = " + str(j+1) + "Epoch = " + str(epoch))
    one_hot_encoded = np.tile(np.array([1,0]),(batch_size,1))
    for i in range(0,len(cat_train_data),batch_size):
        x = cat_train_data[i:i+batch_size]
        
        train(x_train = x, y_train = one_hot_encoded)
    
    one_hot_encoded = np.tile(np.array([0,1]),(batch_size,1))
    for i in range(0,len(dog_train_data),batch_size):
        x = dog_train_data[i:i+batch_size]
        

        train(x_train = x, y_train = one_hot_encoded)
print("training ended")

with open('layer_parameters.pkl', 'wb') as f:
    pickle.dump({
        'layer1_weights': layer1.weights,
        'layer1_biases': layer1.biases,
        'layer2_weights': layer2.weights,
        'layer2_biases': layer2.biases,
        'layer3_weights': layer3.weights,
        'layer3_biases': layer3.biases,
        'layer4_weights': layer4.weights,
        'layer4_biases': layer4.biases,
        'layer5_weights': layer5.weights,
        'layer5_biases': layer5.biases,
        'output_weights': output.weights,
        'output_biases': output.biases
    }, f)


del cat_train_data
del dog_train_data
gc.collect()

zeroarray = np.zeros((max_height,max_width,3),dtype=np.float16)

cat_test = []


print("cat_test_photos translated to numpy array ended")
for image in cat_test_photos:
    img = Image.open(image)
    img = np.array(img)
    
    padded = zeroarray.copy()
    padded[:img.shape[0], : img.shape[1], : ] = img
    padded /= 255.0
    cat_test.append(np.array(padded.flatten()))
print("cat_test_photos translated to numpy array ended")
print("dog_test_photos translated to numpy array ended")


cat_test_data = np.array(cat_test)

del cat_test

del cat_test_photos
gc.collect()

print("cat test will start")
data_acc = []
data_y = []
for i in range(0,len(cat_test_data)):
    x = cat_test_data[i]
    z1 = layer1.forward(x)
    z2 = layer2.forward(z1)
    z3 = layer3.forward(z2)
    z4 = layer4.forward(z3)
    z5 = layer5.forward(z4)
    z6 = np.dot(z5,output.weights) + output.biases
    preds = softmax(z6)
    
    data_acc.append(np.argmax(preds))
    data_y.append(0)
    print("Predicted = " + str(np.argmax(preds)) + " Actual = cat" )
    print("Accuracy = ",acc_calc(np.array(data_acc),np.array(data_y)))

print("cat test ended")
del cat_test_data
gc.collect()
print("dog test will start")

dog_test = []
for image in dog_test_photos:
    img = Image.open(image)
    img = np.array(img)
    
    padded = zeroarray.copy()
    padded[:img.shape[0], : img.shape[1], : ] = img
    padded /= 255.0
    dog_test.append(np.array(padded.flatten()))
print("dog_test_photos translated to numpy array ended")
dog_test_data = np.array(dog_test)

del dog_test
del dog_test_photos
gc.collect()

data_acc = []
data_y = []
for i in range(0,len(dog_test_data)):
    x = dog_test_data[i]
    z1 = layer1.forward(x)
    z2 = layer2.forward(z1)
    z3 = layer3.forward(z2)
    z4 = layer4.forward(z3)
    z5 = layer5.forward(z4)
    z6 = np.dot(z5,output.weights) + output.biases
    preds = softmax(z6)
    
    data_acc.append(np.argmax(preds))
    data_y.append(1)
    print("Predicted = " + str(np.argmax(preds)) + " Actual = dog" )
    print("Accuracy = ",acc_calc(np.array(data_acc),np.array(data_y)))
print("dog test ended")


del dog_test_data
gc.collect()