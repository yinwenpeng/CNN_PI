"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import cPickle
import gzip
import os
import sys
import time

import numpy
import theano.sandbox.neighbours as TSN
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from logistic_sgd_test import read_data
from mlp import HiddenLayer
from WPDefined import ConvFoldPoolLayer,Conv_DynamicK_PoolLayer, dropout_from_layer
from collections import OrderedDict

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), k=4):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)
        #images2neibs produces a 2D matrix
        neighborsForPooling = TSN.images2neibs(ten4=conv_out, neib_shape=(1,conv_out.shape[3]), mode='ignore_borders')

        #k = poolsize[1]

        neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-k:]
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1)

        ii = T.repeat(T.arange(neighborsForPooling.shape[0]), k)
        jj = kNeighborsArgSorted.flatten()
        pooledkmaxTmp = neighborsForPooling[ii, jj]

        # reshape pooledkmaxTmp
        new_shape = T.cast(T.join(0, conv_out.shape[:-2],
                           T.as_tensor([conv_out.shape[2]]),
                           T.as_tensor([k])),
                           'int64')
        pooled_out = T.reshape(pooledkmaxTmp, new_shape, ndim=4)
        
        # downsample each feature map individually, using maxpooling
        '''
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)
        '''
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=1.0, n_epochs=2000,
                    dataset='mnist.pkl.gz',
                    nkerns=[6, 12], batch_size=80):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    root="/mounts/data/proj/wenpeng/Dataset/StanfordSentiment/stanfordSentimentTreebank/"
    embeddingPath='/mounts/data/proj/wenpeng/Downloads/hlbl-embeddings-original.EMBEDDING_SIZE=50.txt'
    embeddingPath2='/mounts/data/proj/wenpeng/MC/src/released_embedding.txt'
    rng = numpy.random.RandomState(23455)
    datasets, embedding_size, embeddings=read_data(root+'5classes/train.txt', root+'5classes/dev.txt', root+'5classes/test.txt', embeddingPath,60)
    #datasets, embedding_size, embeddings=read_data(root+'2classes/train.txt', root+'2classes/dev.txt', root+'2classes/test.txt', embeddingPath,60)

    #datasets = load_data(dataset)
    indices_train, trainY, trainLengths = datasets[0]
    indices_dev, devY, devLengths = datasets[1]
    indices_test, testY, testLengths = datasets[2]
    n_train_batches=indices_train.shape[0]/batch_size
    n_valid_batches=indices_dev.shape[0]/batch_size
    n_test_batches=indices_test.shape[0]/batch_size

    indices_train_theano=theano.shared(numpy.asarray(indices_train, dtype=theano.config.floatX), borrow=True)
    indices_dev_theano=theano.shared(numpy.asarray(indices_dev, dtype=theano.config.floatX), borrow=True)
    indices_test_theano=theano.shared(numpy.asarray(indices_test, dtype=theano.config.floatX), borrow=True)
    indices_train_theano=T.cast(indices_train_theano, 'int32')
    indices_dev_theano=T.cast(indices_dev_theano, 'int32')
    indices_test_theano=T.cast(indices_test_theano, 'int32')
    '''
    indices_train_theano=theano.shared(indices_train, borrow=True)
    indices_dev_theano=theano.shared(indices_dev, borrow=True)
    indices_test_theano=theano.shared(indices_test, borrow=True)
    '''
    

    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x_index = T.imatrix('x_index')   # now, x is the index matrix, must be integer
    y = T.ivector('y')  
    z = T.ivector('z')

    x=embeddings[x_index.flatten()].flatten().reshape((batch_size,60*embedding_size))
    ishape = (embedding_size, 60)  # this is the size of MNIST images
    filter_size1=(1,10)
    filter_size2=(1,7)
    poolsize1=(1, ishape[1]-filter_size1[1]+1)
    
    kmax=30 # this can not be too small, like 20
    ktop=5
    poolsize2=(1, kmax-filter_size2[1]+1) #(1,6)
    lengths=T.maximum(ktop,z/2+1)  # dynamic k-max pooling
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, ishape[0], ishape[1]))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    '''
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_size1[0], filter_size1[1]), poolsize=poolsize1, k=kmax)
    '''
    layer0 = Conv_DynamicK_PoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_size1[0], filter_size1[1]), poolsize=poolsize1, k=lengths, unifiedWidth=kmax)
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    '''
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], ishape[0], kmax),
            filter_shape=(nkerns[1], nkerns[0], filter_size2[0], filter_size2[1]), poolsize=poolsize2, k=ktop)
    '''
    layer1 = ConvFoldPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], ishape[0], kmax),
            filter_shape=(nkerns[1], nkerns[0], filter_size2[0], filter_size2[1]), poolsize=poolsize2, k=ktop)
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer, the output of layers has nkerns[1]=50 images, each is 4*4 size
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * (embedding_size/2) * ktop,
                         n_out=50, activation=T.tanh)
    dropout=dropout_from_layer(rng, layer2.output, 0.5)
    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=dropout, n_in=50, n_out=5)

    # the cost we minimize during training is the NLL of the model
    L1_reg= abs(layer3.W).sum() + abs(layer2.W).sum() +abs(layer1.W).sum()+abs(layer0.W).sum()+abs(embeddings).sum()
    L2_reg = (layer3.W** 2).sum() + (layer2.W** 2).sum()+ (layer1.W** 2).sum()+(layer0.W** 2).sum()+(embeddings**2).sum()
    cost = layer3.negative_log_likelihood(y)+0*L1_reg+0.0000001*L2_reg
    
    #cost = layer3.negative_log_likelihood(y)
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x_index: indices_test_theano[index * batch_size: (index + 1) * batch_size],
                y: testY[index * batch_size: (index + 1) * batch_size],
                z: testLengths[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x_index: indices_dev_theano[index * batch_size: (index + 1) * batch_size],
                y: devY[index * batch_size: (index + 1) * batch_size],
                z: devLengths[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params+[embeddings]
    
    accumulator=[]
    for para_i in params:
        eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
      
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    grads[len(grads)-1]=T.set_subtensor(grads[len(grads)-1][0], theano.shared(numpy.zeros(embedding_size)))

    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / (T.sqrt(acc)+ 1e-5)))
        updates.append((acc_i, acc))   
       
    train_model = theano.function([index], cost, updates=updates,
          givens={
            x_index: indices_train_theano[index * batch_size: (index + 1) * batch_size],
            y: trainY[index * batch_size: (index + 1) * batch_size],
            z: trainLengths[index * batch_size: (index + 1) * batch_size] })

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches): # each batch
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))


            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
