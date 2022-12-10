import numpy
import random
import time
import struct
import requests
import os
import gzip

def download_mnist_data(filename):
    print('downloading', filename)
    os.makedirs('data', exist_ok=True)
    r = requests.get('http://yann.lecun.com/exdb/mnist/' + filename)
    if r.status_code != 200:
        return False
    open('data/' + filename, 'wb').write(r.content)
    return True

def decompress_mnist_data(filename):
    print('decompressing', filename)
    with gzip.open(f'data/{filename}', 'rb') as fc:
        with open(f"data/{filename.split('.')[0]}", 'wb') as fd:
            fd.write(fc.read())
    return True

def get_mnist_data():
    filenames = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    for filename in filenames:
        if os.path.exists(f'data/{filename}') or download_mnist_data(filename):
            if not os.path.exists(f"data/{filename.split('.')[0]}"):
                decompress_mnist_data(filename)
        else:
            return False
    return True

def read(dataset= 'training', path= './data'):
    if dataset == 'training':
        fname_img = path + '/train-images-idx3-ubyte'
        fname_lbl = path + '/train-labels-idx1-ubyte'
    elif dataset == 'testing':
        fname_img = path + '/t10k-images-idx3-ubyte'
        fname_lbl = path + '/t10k-labels-idx1-ubyte'
    else:
        raise ValueError('dataset must be \'testing\' or \'training\'')
    flbl = open(fname_lbl, 'rb')
    magic_no, size = struct.unpack('>II', flbl.read(8))
    if magic_no != 2049:
        raise RuntimeError('Invalid MNIST ' + dataset + ' label file')
    fimg = open(fname_img, 'rb')
    magic_no, size, rows, cols = struct.unpack('>IIII', fimg.read(16))
    if magic_no != 2051:
        raise RuntimeError('Invalid MNIST ' + dataset + ' image file')
    img_size = rows * cols
    data = []
    for lbl in flbl.read():
        x = numpy.frombuffer(fimg.read(img_size), dtype=numpy.uint8).astype(numpy.float64) / 255
        y = numpy.zeros(10)
        y[lbl] = 1
        data.append([x.reshape((1, img_size)), y.reshape((1, 10))])
    flbl.close()
    fimg.close()
    return data

def sigmoid(z):
    return 1.0/(1.0+numpy.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class Layer:
    def __init__(self, inp_size, out_size, ETA):
        self.ETA = ETA
        self.b = numpy.random.randn(1, out_size)
        self.w = numpy.random.randn(inp_size, out_size)
        self.bgrad = 0.0
        self.wgrad = 0.0
    def forward(self, inp):
        self.x = inp
        self.y = numpy.dot(self.x, self.w) + self.b
        self.a = sigmoid(self.y)
        return self.a
    def compgrad(self, a_grad):
        y_grad = sigmoid_prime(self.y) * a_grad
        self.bgrad += y_grad
        self.wgrad += numpy.dot(self.x.T, y_grad)
        return numpy.dot(y_grad, self.w.T)
    def backprop(self):
        self.w -= self.ETA * self.wgrad
        self.b -= self.ETA * self.bgrad
        self.bgrad = 0
        self.wgrad = 0

class Model:
    def __init__(self, net_size, ETA):
        self.layers = []
        for i in range(len(net_size) - 1):
            self.layers.append(Layer(net_size[i], net_size[i + 1], ETA))
    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x
    def compgrad(self, a_grad):
        for l in reversed(self.layers):
            a_grad = l.compgrad(a_grad)
    def backprop(self):
        for l in reversed(self.layers):
            l.backprop()

def train(traindata, BATCH_SIZE= 10):
    data_size = len(traindata)
    random.shuffle(traindata)
    batches = (traindata[i : i + BATCH_SIZE] for i in range(0, data_size, BATCH_SIZE))
    loss = 0.0
    for batch in batches:
        for x, y in batch:
            a = network.forward(x)
            loss += (a - y) ** 2 / BATCH_SIZE
            a_grad = 2 * (a - y) / BATCH_SIZE
            network.compgrad(a_grad)
        network.backprop()
    return numpy.sum(loss) * BATCH_SIZE / data_size

def test(testdata):
    correct = 0
    for x,y in testdata:
        n = network.forward(x)
        if numpy.array_equal(numpy.around(n), y):
            correct += 1
    return correct

EPOCHS = 15
BATCH_SIZE = 10
ETA = 1.5
if get_mnist_data():
    print('Training and Testing every Epoch . . .')
    traindata = read()
    testdata = read(dataset='testing')
    network = Model([784, 39, 10], ETA)
    for itr in range(EPOCHS):
        print(f'Loss: {train(traindata, BATCH_SIZE)} Epoch: {itr} ( {test(testdata)} / {len(testdata)} )')
else:
    print('Unable to download MNIST dataset')