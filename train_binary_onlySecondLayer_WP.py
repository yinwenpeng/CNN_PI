'''
learning_rate=0.1, n_epochs=2000, nkerns=[6, 14], batch_size=70, useAllSamples=0, kmax=30, ktop=4, filter_size=[7,5], hidden_units=50, L2_weight=0.000005, dropout_p=0.2, useEmb=1, performance=20.604396%

'''
import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(2000)
import time

import numpy
import theano.sandbox.neighbours as TSN
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from WPDefined import ConvFoldPoolLayer,Conv_DynamicK_PoolLayer, dropout_from_layer, shared_dataset, read_data_WP
from collections import OrderedDict


def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, nkerns=[6, 14], batch_size=70, useAllSamples=0, kmax=30, ktop=4, filter_size=[7,5],
                    hidden_units=50, L2_weight=0.000005, dropout_p=0.2, useEmb=1):
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
    rng = numpy.random.RandomState(99999)
    datasets, embedding_size, embeddings=read_data_WP(root+'2classes/train.txt', root+'2classes/dev.txt', root+'2classes/test.txt', embeddingPath,60, useEmb)
    #datasets, embedding_size, embeddings=read_data(root+'2classes/train.txt', root+'2classes/dev.txt', root+'2classes/test.txt', embeddingPath,60)

    #datasets = load_data(dataset)
    indices_train, trainY, trainLengths, trainLeftPad, trainRightPad= datasets[0]
    indices_dev, devY, devLengths, devLeftPad, devRightPad= datasets[1]
    indices_test, testY, testLengths, testLeftPad, testRightPad= datasets[2]
    n_train_batches=indices_train.shape[0]/batch_size
    n_valid_batches=indices_dev.shape[0]/batch_size
    n_test_batches=indices_test.shape[0]/batch_size
    
    train_batch_start=[]
    dev_batch_start=[]
    test_batch_start=[]
    if useAllSamples:
        train_batch_start=list(numpy.arange(n_train_batches)*batch_size)+[indices_train.shape[0]-batch_size]
        dev_batch_start=list(numpy.arange(n_valid_batches)*batch_size)+[indices_dev.shape[0]-batch_size]
        test_batch_start=list(numpy.arange(n_test_batches)*batch_size)+[indices_test.shape[0]-batch_size]
        n_train_batches=n_train_batches+1
        n_valid_batches=n_valid_batches+1
        n_test_batches=n_test_batches+1
    else:
        train_batch_start=list(numpy.arange(n_train_batches)*batch_size)
        dev_batch_start=list(numpy.arange(n_valid_batches)*batch_size)
        test_batch_start=list(numpy.arange(n_test_batches)*batch_size)

    indices_train_theano=theano.shared(numpy.asarray(indices_train, dtype=theano.config.floatX), borrow=True)
    indices_dev_theano=theano.shared(numpy.asarray(indices_dev, dtype=theano.config.floatX), borrow=True)
    indices_test_theano=theano.shared(numpy.asarray(indices_test, dtype=theano.config.floatX), borrow=True)
    indices_train_theano=T.cast(indices_train_theano, 'int32')
    indices_dev_theano=T.cast(indices_dev_theano, 'int32')
    indices_test_theano=T.cast(indices_test_theano, 'int32')
    
    

    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x_index = T.imatrix('x_index')   # now, x is the index matrix, must be integer
    y = T.ivector('y')  
    #z = T.ivector('z')
    left=T.ivector('left')
    right=T.ivector('right')
    
    x=embeddings[x_index.flatten()].reshape((batch_size,60, embedding_size)).transpose(0, 2, 1).flatten()
    ishape = (embedding_size, 60)  # this is the size of MNIST images
    filter_size1=(1,filter_size[0])
    filter_size2=(1,filter_size[1])
    #poolsize1=(1, ishape[1]-filter_size1[1]+1) #?????????????????????????????
    poolsize1=(1, ishape[1]+filter_size1[1]-1)


    left_after_conv=left
    right_after_conv=right
    
    #kmax=30 # this can not be too small, like 20
    #ktop=6
    #poolsize2=(1, kmax-filter_size2[1]+1) #(1,6)
    poolsize2=(1, kmax+filter_size2[1]-1) #(1,6)
    #dynamic_lengths=T.maximum(ktop,z/2+1)  # dynamic k-max pooling
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, ishape[0], ishape[1]))
    
    layer1 = ConvFoldPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_size1[0], filter_size1[1]), poolsize=poolsize1, k=ktop, left=left_after_conv, right=right_after_conv)
    
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer, the output of layers has nkerns[1]=50 images, each is 4*4 size
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[0] * (embedding_size/2) * ktop,
                         n_out=hidden_units, activation=T.tanh)
    
    dropout=dropout_from_layer(rng, layer2.output, dropout_p)               #dropout
    layer3 = LogisticRegression(rng, input=dropout, n_in=hidden_units, n_out=2)
    
    #layer3 = LogisticRegression(rng, input=layer2.output, n_in=50, n_out=2)
    # the cost we minimize during training is the NLL of the model
    #L1_reg= abs(layer3.W).sum() + abs(layer2.W).sum() +abs(layer1.W).sum()+abs(layer0.W).sum()+abs(embeddings).sum()
    L2_reg = (layer3.W** 2).sum() + (layer2.W** 2).sum()+ (layer1.W** 2).sum()+(embeddings**2).sum()
    #L2_reg = (layer3.W** 2).sum() + (layer2.W** 2).sum()+(layer0.W** 2).sum()+(embeddings**2).sum()
    #cost must have L2, otherwise, will produce nan, while with L2, each word embedding will be updated
    cost = layer3.negative_log_likelihood(y)+L2_weight*L2_reg
    
    #cost = layer3.negative_log_likelihood(y)
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x_index: indices_test_theano[index: index + batch_size],
                y: testY[index: index + batch_size],
                left: testLeftPad[index: index + batch_size],
                right: testRightPad[index: index + batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x_index: indices_dev_theano[index: index + batch_size],
                y: devY[index: index + batch_size],
                left: devLeftPad[index: index + batch_size],
                right: devRightPad[index: index + batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params +[embeddings]
    #params = layer3.params + layer2.params + layer0.params+[embeddings]
    
    accumulator=[]
    for para_i in params:
        eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
      
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.

    '''   
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))
    
    '''
    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        acc = acc_i + T.sqr(grad_i)
        if param_i == embeddings:
            updates.append((param_i, T.set_subtensor((param_i - learning_rate * grad_i / T.sqrt(acc))[0], theano.shared(numpy.zeros(embedding_size)))))   #AdaGrad
        else:
            updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc)))   #AdaGrad
        updates.append((acc_i, acc))    
       
    train_model = theano.function([index], [cost,layer3.errors(y)], updates=updates,
          givens={
            x_index: indices_train_theano[index: index + batch_size],
            y: trainY[index: index + batch_size],
            left: trainLeftPad[index: index + batch_size],
            right: trainRightPad[index: index + batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 50000  # look as this many examples regardless
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
        #for minibatch_index in xrange(n_train_batches): # each batch
        minibatch_index=0
        for batch_start in train_batch_start: 
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + minibatch_index +1
            minibatch_index=minibatch_index+1
            
            cost_ij, error_ij = train_model(batch_start)
            if iter % n_train_batches == 0:
                print 'training @ iter = '+str(iter)+' cost: '+str(cost_ij)+' error: '+str(error_ij)
            if iter % validation_frequency == 0:

                # compute zero-one loss on validation set
                #validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                validation_losses = [validate_model(i) for i in dev_batch_start]
                this_validation_loss = numpy.mean(validation_losses)
                print('\t\tepoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index , n_train_batches, \
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
                    test_losses = [test_model(i) for i in test_batch_start]
                    test_score = numpy.mean(test_losses)
                    print(('\t\t\t\tepoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index, n_train_batches,
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