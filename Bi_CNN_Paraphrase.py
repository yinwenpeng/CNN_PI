
import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy
import theano
import theano.tensor as T
import theano.sandbox.neighbours as TSN

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from WPDefined import ConvFoldPoolLayer, dropout_from_layer, shared_dataset, repeat_whole_matrix
from cis.deep.utils.theano import debug_print
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv



def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, nkerns=[5, 3, 3], batch_size=10, useAllSamples=0, kmax=35, ktop=8, filter_size=[3,5, 5],
                    L2_weight=0.00005, dropout_p=0.8, useEmb=1, task=2, dataMode=2, maxSentLength=60, train_lines=4070, emb_size=4, sentEm_length=100,
                    Np=[15, 15, 4]):
#def evaluate_lenet5(learning_rate=0.1, n_epochs=2000, nkerns=[6, 12], batch_size=70, useAllSamples=0, kmax=30, ktop=5, filter_size=[10,7],
#                    L2_weight=0.000005, dropout_p=0.5, useEmb=0, task=5, corpus=1):

    rootPath="/mounts/Users/student/wenpeng/workspace/phraseEmbedding/MicrosoftParaphrase/tokenized_msr/";
    embeddingPath='/mounts/data/proj/wenpeng/Downloads/hlbl-embeddings-original.EMBEDDING_SIZE=50.txt'
    embeddingPath2='/mounts/data/proj/wenpeng/MC/src/released_embedding.txt'
    rng = numpy.random.RandomState(23455)
    datasets, embedding_size, embeddings=load_MSR_corpus(rootPath+'tokenized_train.txt', rootPath+'tokenized_test.txt', embeddingPath,maxSentLength, useEmb, dataMode, train_lines, emb_size)
    #datasets, embedding_size, embeddings=read_data(root+'2classes/train.txt', root+'2classes/dev.txt', root+'2classes/test.txt', embeddingPath,60)

    #datasets = load_data(dataset)
    indices_train, trainY, trainLengths, trainLeftPad, trainRightPad= datasets[0]
    #print trainY.eval().shape[0]
    indices_dev, devY, devLengths, devLeftPad, devRightPad= datasets[1]
    indices_test, testY, testLengths, testLeftPad, testRightPad= datasets[2]
    n_train_batches=indices_train.shape[0]/(batch_size*2)
    n_valid_batches=indices_dev.shape[0]/(batch_size*2)
    n_test_batches=indices_test.shape[0]/(batch_size*2)
    remain_train=indices_train.shape[0]%(batch_size*2)
    
    train_batch_start=[]
    dev_batch_start=[]
    test_batch_start=[]
    if useAllSamples:
        train_batch_start=list(numpy.arange(n_train_batches)*(batch_size*2))+[indices_train.shape[0]-(batch_size*2)]
        dev_batch_start=list(numpy.arange(n_valid_batches)*(batch_size*2))+[indices_dev.shape[0]-(batch_size*2)]
        test_batch_start=list(numpy.arange(n_test_batches)*(batch_size*2))+[indices_test.shape[0]-(batch_size*2)]
        n_train_batches=n_train_batches+1
        n_valid_batches=n_valid_batches+1
        n_test_batches=n_test_batches+1
    else:
        train_batch_start=list(numpy.arange(n_train_batches)*(batch_size*2))
        dev_batch_start=list(numpy.arange(n_valid_batches)*(batch_size*2))
        test_batch_start=list(numpy.arange(n_test_batches)*(batch_size*2))

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
    
    x=embeddings[x_index.flatten()].reshape(((batch_size*2),maxSentLength, embedding_size)).transpose(0, 2, 1).flatten()
    ishape = (embedding_size, maxSentLength)  # this is the size of MNIST images
    filter_size1=(embedding_size,filter_size[0])
    #filter_size2=(embedding_size/2,filter_size[1])
    #poolsize1=(1, ishape[1]-filter_size1[1]+1) #?????????????????????????????
    poolsize1=(1, ishape[1]+filter_size1[1]-1)

    '''
    left_after_conv=T.maximum(0,left-filter_size1[1]+1)
    right_after_conv=T.maximum(0, right-filter_size1[1]+1)
    '''
    left_after_conv=left
    right_after_conv=right
    
    #kmax=30 # this can not be too small, like 20
    #ktop=6
    #poolsize2=(1, kmax-filter_size2[1]+1) #(1,6)
    #poolsize2=(1, kmax+filter_size2[1]-1) #(1,6)
    #dynamic_lengths=T.maximum(ktop,z/2+1)  # dynamic k-max pooling
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape(((batch_size*2), 1, ishape[0], ishape[1]))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    '''
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_size1[0], filter_size1[1]), poolsize=poolsize1, k=kmax)
    '''
    '''
    layer0 = Conv_Fold_DynamicK_PoolLayer(rng, input=layer0_input,
            image_shape=((batch_size*2), 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_size1[0], filter_size1[1]), poolsize=poolsize1, k=dynamic_lengths, unifiedWidth=kmax, left=left_after_conv, right=right_after_conv, firstLayer=True)
    '''
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    '''
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], ishape[0], kmax),
            filter_shape=(nkerns[1], nkerns[0], filter_size2[0], filter_size2[1]), poolsize=poolsize2, k=ktop)
    '''
    '''
    left_after_conv=T.maximum(0, layer0.leftPad-filter_size2[1]+1)
    right_after_conv=T.maximum(0, layer0.rightPad-filter_size2[1]+1)
    '''
    #left_after_conv=layer0.leftPad
    #right_after_conv=layer0.rightPad
    dynamic_lengths=T.repeat([ktop],(batch_size*2))  # dynamic k-max pooling
    #layer0_output = debug_print(layer0.output, 'layer0.output')
    '''
    layer1 = ConvFoldPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], ishape[0]/2, kmax),
            filter_shape=(nkerns[1], nkerns[0], filter_size2[0], filter_size2[1]), poolsize=poolsize2, k=ktop, left=left_after_conv, right=right_after_conv)
    '''
    '''
    layer1 = Conv_Fold_DynamicK_PoolLayer(rng, input=layer0.output,
            image_shape=((batch_size*2), nkerns[0], ishape[0]/2, kmax),
            filter_shape=(nkerns[1], nkerns[0], filter_size2[0], filter_size2[1]), poolsize=poolsize2, k=dynamic_lengths, unifiedWidth=ktop, left=left_after_conv, right=right_after_conv, firstLayer=False)    
    '''
    
    layer0 = Conv_Fold_DynamicK_PoolLayer(rng, input=layer0_input,
            image_shape=((batch_size*2), 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_size1[0], filter_size1[1]), poolsize=poolsize1, k=dynamic_lengths, unifiedWidth=ktop, left=left_after_conv, right=right_after_conv, firstLayer=False)    
    
    '''
    poolsize3=(6,8)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=((batch_size*2), 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_size[2], filter_size[2]), poolsize=poolsize3)    
    '''     
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    
    uni_simi=compute_simi_feature(layer0.input, maxSentLength)
    fold_out=debug_print(layer0.fold_out,'fold_out')
    ngram_simi=compute_simi_feature(fold_out, poolsize1[1])
    pool_out=debug_print(layer0.output, 'pool_out')
    sent_simi=compute_simi_feature(pool_out, ktop)
    
    unigram_matrices=unified_similarity_matrix(uni_simi, left, right, batch_size, maxSentLength, Np[0])
    ngram_matrices=unified_similarity_matrix(ngram_simi, left, right, batch_size, poolsize1[1], Np[1])
    pads=T.repeat([0],(batch_size*2)) 
    sent_matrices=unified_similarity_matrix(sent_simi, pads, pads, batch_size, ktop, Np[2])
    
    all_simi=T.concatenate([unigram_matrices.flatten(2), ngram_matrices.flatten(2), sent_matrices.flatten(2)], axis=1)
    
    '''
    
    
    
    
    
    layer2_input = layer0.output.flatten(2)
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[0] * (embedding_size/2) * ktop, n_out=sentEm_length, activation=T.tanh)
    
    
    layer2_input=Vector_Corelation(layer2.output) #now, its output is a (batch_size, layer2_input.shape[1], layer2_input.shape[1])
    #ishape1=((embedding_size+filter_size[2]-1)/poolsize3[0])*((maxSentLength+filter_size[2]-1)/poolsize3[1])*nkerns[0]
    '''
    '''
    layer1_input = layer0.output.flatten(2)
    
    
    ishape1=(embedding_size+filter_size[2]-1)*(maxSentLength+filter_size[2]-1)*nkerns[0]
    layer1_input=Vector_Corelation(layer1_input) 
    layer1 = LeNetConvPoolLayer_valid(rng, input=layer1_input,
            image_shape=(batch_size, 1, ishape1, ishape1),
            filter_shape=(1, 1, 3, 3), poolsize=(1,1))    
    #dropout=dropout_from_layer(rng, layer2_input, dropout_p)
    # construct a fully-connected sigmoidal layer, the output of layers has nkerns[1]=50 images, each is 4*4 size
    #layer2 = FullyConnectedLayer(rng, input=dropout, n_in=nkerns[1] * (embedding_size/4) * ktop, n_out=task)
    layer2_input = layer1.output.flatten(2)
    '''
    
    #layer2_input=1-layer2_input
    layer4 = LogisticRegression(rng, input=all_simi, n_in=Np[0]**2+nkerns[0]*(Np[1]**2)+nkerns[0]*(Np[2]**2), n_out=task)
    #layer3=SoftMaxlayer(input=layer2.output)
    #layer3 = LogisticRegression(rng, input=layer2.output, n_in=50, n_out=2)
    # the cost we minimize during training is the NLL of the model
    #L1_reg= abs(layer3.W).sum() + abs(layer2.W).sum() +abs(layer1.W).sum()+abs(layer0.W).sum()+abs(embeddings).sum()
    L2_reg =(layer4.W** 2).sum()+(layer0.W** 2).sum()+(embeddings**2).sum()
    #L2_reg = (layer3.W** 2).sum() + (layer2.W** 2).sum()+(layer0.W** 2).sum()+(embeddings**2).sum()
    #cost must have L2, otherwise, will produce nan, while with L2, each word embedding will be updated
    cost =layer4.negative_log_likelihood(y)+L2_weight*L2_reg
    
    #cost = layer3.negative_log_likelihood(y)
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer4.errors(y),
             givens={
                x_index: indices_test_theano[index: index + (batch_size*2)],
                y: testY[(index/2): (index/2) + batch_size],
                left: testLeftPad[index: index + (batch_size*2)],
                right: testRightPad[index: index + (batch_size*2)]})
    '''
    validate_model = theano.function([index], layer2.errors(y),
            givens={
                x_index: indices_dev_theano[index: index + (batch_size*2)],
                y: devY[(index/2): (index/2) + batch_size]})
    '''
    # create a list of all model parameters to be fit by gradient descent
    params = layer4.params   +layer0.params +[embeddings]
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
        if param_i == embeddings:
            updates.append((param_i, T.set_subtensor((param_i - learning_rate * grad_i)[0], theano.shared(numpy.zeros(embedding_size)))))   #AdaGrad
        else:
            updates.append((param_i, param_i - learning_rate * grad_i))   #AdaGrad
 
    
    '''
    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        grad_i=debug_print(grad_i,'grad_i')
        acc = acc_i + T.sqr(grad_i)
        if param_i == embeddings:
            updates.append((param_i, T.set_subtensor((param_i - learning_rate * grad_i / T.sqrt(acc))[0], theano.shared(numpy.zeros(embedding_size)))))   #AdaGrad
        else:
            updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc)))   #AdaGrad
        updates.append((acc_i, acc))    
  
    train_model = theano.function([index], [cost,layer4.errors(y)], updates=updates,
          givens={
            x_index: indices_train_theano[index: index + (batch_size*2)],
            y: trainY[(index/2): (index/2) + batch_size],
            left: trainLeftPad[index: index + (batch_size*2)],
            right: trainRightPad[index: index + (batch_size*2)]})

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
    validation_frequency = min(n_train_batches/5, patience / 2)
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
            #if epoch %2 ==0:
            #    batch_start=batch_start+remain_train
            time.sleep(0.5)
            cost_ij, error_ij = train_model(batch_start)
            
            if iter % n_train_batches == 0:
                print 'training @ iter = '+str(iter)+' cost: '+str(cost_ij)+' error: '+str(error_ij)
            #if iter ==1:
            #    exit(0)
            
            if iter % validation_frequency == 0:
                test_losses = [test_model(i) for i in test_batch_start]
                test_score = numpy.mean(test_losses)
                print(('\t\t\t\tepoch %i, minibatch %i/%i, test error of best '
                           'model %f %%') %
                          (epoch, minibatch_index, n_train_batches,
                           test_score * 100.))
                '''
                #print 'validating & testing...'
                # compute zero-one loss on validation set
                validation_losses = []
                for i in dev_batch_start:
                    time.sleep(0.5)
                    validation_losses.append(validate_model(i))
                #validation_losses = [validate_model(i) for i in dev_batch_start]
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
            '''

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


def unify_eachone(tensor, left1, right1, left2, right2, dim, Np):

    #tensor: (1, feature maps, 66, 66)
    sentlength_1=dim-left1-right1
    sentlength_2=dim-left2-right2
    core=tensor[:,:, left1:(dim-right1),left2:(dim-right2) ]

    repeat_row=Np/sentlength_1
    extra_row=Np%sentlength_1
    repeat_col=Np/sentlength_2
    extra_col=Np%sentlength_2    

    #repeat core
    core_1=repeat_whole_matrix(core, 5, True) 
    core_2=repeat_whole_matrix(core_1, 5, False)
    
    new_rows=T.maximum(sentlength_1, sentlength_1*repeat_row+extra_row)
    new_cols=T.maximum(sentlength_2, sentlength_2*repeat_col+extra_col)
    
    core=debug_print(core_2[:,:, :new_rows, : new_cols],'core')
    #determine x, y start positions
    size_row=new_rows/Np
    remain_row=new_rows%Np
    size_col=new_cols/Np
    remain_col=new_cols%Np
    
    xx=debug_print(T.concatenate([T.arange(Np-remain_row+1)*size_row, (Np-remain_row)*size_row+(T.arange(remain_row)+1)*(size_row+1)]),'xx')
    yy=debug_print(T.concatenate([T.arange(Np-remain_col+1)*size_col, (Np-remain_col)*size_col+(T.arange(remain_col)+1)*(size_col+1)]),'yy')
    
    list_of_maxs=[]
    for i in xrange(Np):
        for j in xrange(Np):
            region=debug_print(core[:,:, xx[i]:xx[i+1], yy[j]:yy[j+1]],'region')
            maxvalue1=debug_print(T.max(region, axis=2), 'maxvalue1')
            maxvalue=debug_print(T.max(maxvalue1, axis=2).transpose(), 'maxvalue')
            list_of_maxs.append(maxvalue)
    

    all_max_value=T.concatenate(list_of_maxs, axis=1).reshape((tensor.shape[0], tensor.shape[1], Np, Np))
    
    return all_max_value
def unified_similarity_matrix(tensor, left, right, batch_size, dim, Np):
    # tensor: (batch_size, feature maps, 66, 66)
    list_of_matrices=[]
    for i in range(batch_size):
        list_of_matrices.append(unify_eachone(tensor[i:(i+1)], left[i*2], right[i*2], left[i*2+1], right[i*2+1], dim, Np))
    pooled_out=T.concatenate(list_of_matrices, axis=0)
    return pooled_out

def compute_simi_feature(tensor, dim):
    odd_tensor=debug_print(tensor[0:tensor.shape[0]:2,:,:,:],'odd_tensor')
    even_tensor=debug_print(tensor[1:tensor.shape[0]:2,:,:,:], 'even_tensor')

    repeated_1=debug_print(T.repeat(odd_tensor, dim, axis=3),'repeat_odd')
    repeated_2=debug_print(repeat_whole_matrix(even_tensor, dim, False),'repeat_even')
    #repeated_2=T.repeat(even_tensor, even_tensor.shape[3], axis=2).reshape((tensor.shape[0]/2, tensor.shape[1], tensor.shape[2], tensor.shape[3]**2))    

    square_distance=T.sum(T.sqr(repeated_1-repeated_2), axis=2)
    root_square_distance=T.sqrt(T.maximum(square_distance, 1e-20))
    list_of_simi=1.0/T.exp(root_square_distance)
    '''
    length_1=debug_print(T.sqrt(T.sum(T.sqr(repeated_1), axis=2)),'length_1')
    length_2=debug_print(T.sqrt(T.sum(T.sqr(repeated_2), axis=2)), 'length_2')

    multi=debug_print(repeated_1*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=2),'sum_multi')
    
    list_of_simi= debug_print(sum_multi/(length_1*length_2+1e-20),'list_of_simi')   #to get rid of zero length
    '''
    return list_of_simi.reshape((tensor.shape[0]/2, tensor.shape[1], tensor.shape[3], tensor.shape[3]))
'''
def vector_similarities(matrix_1, matrix_2):
    repeated_1=T.repeat(matrix_1, matrix_2.shape[1], axis=1)
    repeated_2=T.repeat(matrix_2, matrix_1.shape[1], axis=0).reshape((matrix_2.shape[0],))
    
    length_1=T.sqrt(T.sum(T.sqr(repeated_1), axis=0))
    length_2=T.sqrt(T.sum(T.sqr(repeated_2), axis=0))
    
    multi=repeated_1*repeated_2
    sum_multi=T.sum(multi, axis=0)
    
    list_of_simi= sum_multi/(length_1*length_2+1e-20)   #to get rid of zero length
    
    return list_of_simi.reshape((matrix_1.shape[1], matrix_2.shape[1]))
'''



def Vector_Corelation(matrix):
    row=matrix.shape[0]
    col=matrix.shape[1]
    odd_matrix=matrix[0:row:2]
    even_matrix=matrix[1:row:2]
    '''
    reshaped_odd_matrix=odd_matrix.reshape((odd_matrix.shape[0], col, 1))
    reshaped_even_matrix=even_matrix.reshape((even_matrix.shape[0], 1, col))
    multiplication=debug_print(reshaped_odd_matrix*reshaped_even_matrix, 'multiplication')
    flattened_multiplication=debug_print(multiplication.flatten(2), 'flattened_multiplication')
    return T.tanh(flattened_multiplication)          #also consider sigma function
    '''
    '''
    abs_error=abs(odd_matrix-even_matrix)
    return abs_error
    '''
    square_error=(odd_matrix-even_matrix)**2
    inverse_square=1/(1+square_error)
    return inverse_square
    '''
    repeated_odd=T.repeat(odd_matrix, col, axis=1)
    reshaped_odd=repeated_odd.reshape((row/2, col, col))
    reshaped_even=even_matrix.reshape((row/2, 1, col))
    abs_error=abs(reshaped_odd-reshaped_even)
    #flattened_error=abs_error.flatten(2)
    return abs_error.reshape((abs_error.shape[0], 1, abs_error.shape[1], abs_error.shape[2]))
    '''
def load_MSR_corpus(trainFile, testFile, emb_file, maxlength, useEmb, dataMode, train_lines, emb_size):
    #first store emb_file into a dict
    embeddings=[]
    #embeddings_Q=[]
    word2id={}
    embedding_size=0
    if useEmb:
        embeddingsFile=open(emb_file,'r')
    
        for num_lines, line in enumerate(embeddingsFile):
            
            #if num_lines > 99:
            #    break
            
            tokens=line.strip().split() # split() with no parameters means seperating using consecutive spaces
            vector=[]
            embedding_size=len(tokens)-1
            for i in range(1, embedding_size+1):
                vector.append(float(tokens[i]))
            if num_lines==0:
                embeddings.append(numpy.zeros(embedding_size)) 
                #embeddings_Q.append(numpy.zeros(embedding_size)) 
            embeddings.append(vector)
            #embeddings_Q.append(vector)
            word2id[tokens[0]]=num_lines+1 # word index starts from 1
    
        embeddingsFile.close()
    else:
        embedding_size=emb_size   #the paper uses 48
        embeddings.append(numpy.zeros(embedding_size)) 
        #embeddings_Q.append(numpy.zeros(embedding_size)) 
    word_count=len(embeddings)
    print 'Totally, '+str(word_count)+' word embeddings.'
    
    def load_train_file(file, embeddings, word_count, word2id, train_lines):   
        #id2count={}
        senti_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        line_count=0
        for line in senti_file:
            line_count=line_count+1
            if line_count > train_lines:
                break
            tokens=line.strip().split('\t')  # label, sent1, sent2
            Y.append(int(tokens[0])) 
            #sent1
            for i in range(1,3):
                sent=[]
                words=tokens[i].strip().lower().split(' ')    # all words converted into lowercase
                length=len(words)
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+tokens[i]
                    exit(0)   
                sent+=[0]*left
                for word in words:
                    #sent.append(word2id.get(word))
                    
                    id=word2id.get(word, -1)   # possibly the embeddings are for words with lowercase
                    if id == -1:
                        embeddings.append(numpy.random.uniform(-1,1,embedding_size)) # generate a random embedding for an unknown word
                        #embeddings_target.append(numpy.random.uniform(-1,1,embedding_size))
                        word2id[word]=word_count
                        #id2count[word_count]=1 #1 means new words
                        sent.append(word_count)
                        word_count=word_count+1                  
                    else:
                        sent.append(id)
                        #id2count[id]=id2count[id]+1# add 1 for kown words
                sent+=[0]*right
                data.append(sent)
        senti_file.close()
        print 'Vocab:'+str(len(word2id))
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad), numpy.array(embeddings), word_count, word2id

    def load_dev_file(file, word_count, word2id, train_lines):
        senti_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        unknown=0
        total=0
        line_count=0
        for line in senti_file:
            line_count=line_count+1
            if line_count <= train_lines:
                continue
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])) # make the label starts from 0 to 4
            for i in range(1,3):
                sent=[]
                words=tokens[i].strip().lower().split(' ')    # all words converted into lowercase
                length=len(words)
                total=total+length
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+line
                    exit(0)  
                sent+=[0]*left
                for word in words:
                    #sent.append(word2id.get(word))
                    
                    id=word2id.get(word, -1)   # possibly the embeddings are for words with lowercase
                    if id == -1:    
                        unknown=unknown+1              
                        #sent.append(numpy.random.random_integers(word_count)) 
                        #sent.append(pre_index) # for new words in dev or test data, let's assume its embedding is zero      
                        sent.append(0)           
                    else:
                        sent.append(id)
                        #pre_index=id
                sent+=[0]*right
                data.append(sent)
        senti_file.close()
        print 'Unknown:'+str(unknown)+' rate:'+str(unknown*1.0/total)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
    def load_dev_file_preIndex(file, word_count, word2id, train_lines):
        senti_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        unknown=0
        total=0
        line_count=0
        for line in senti_file:
            line_count=line_count+1
            if line_count <= train_lines:
                continue            
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])) # make the label starts from 0 to 4
            for i in range(1,3):
                pre_index=0
                sent=[]
                words=tokens[i].strip().lower().split(' ')    # all words converted into lowercase
                length=len(words)
                total=total+length
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+line
                    exit(0)  
                sent+=[0]*left
                for word in words:
                    #sent.append(word2id.get(word))
                    
                    id=word2id.get(word, -1)   # possibly the embeddings are for words with lowercase
                    if id == -1:       
                        unknown=unknown+1           
                        #sent.append(numpy.random.random_integers(word_count)) 
                        sent.append(pre_index) # for new words in dev or test data, let's assume its embedding is zero                
                    else:
                        sent.append(id)
                        pre_index=id
                sent+=[0]*right
                data.append(sent)
        senti_file.close()
        print 'Unknown:'+str(unknown)+' rate:'+str(unknown*1.0/total)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
    def load_dev_file_skipUnknown(file, word_count, word2id, train_lines):
        senti_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        unknown=0
        total=0
        line_count=0
        for line in senti_file:
            line_count=line_count+1
            if line_count <= train_lines:
                continue
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])) # make the label starts from 0 to 4
            for i in range(1,3):
                sent=[]
                words=tokens[i].strip().lower().split(' ')    # all words converted into lowercase
                knownWords=[]
                for word in words:
                    id=word2id.get(word, -1) 
                    if id != -1:
                        knownWords.append(word)
                    else:
                        unknown=unknown+1
                
                length=len(knownWords)
                total=total+length
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+line
                    exit(0)  
                sent+=[0]*left
                for word in knownWords:
                    sent.append(word2id.get(word, -1))
                sent+=[0]*right
                data.append(sent)
        senti_file.close()
        print 'Unknown:'+str(unknown)+' rate:'+str(unknown*1.0/total)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
    
    
    def load_test_file(file, word_count, word2id):
        senti_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        unknown=0
        total=0
        for line in senti_file:
            #pre_index=0
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])) # make the label starts from 0 to 4
            for i in range(1,3):
                sent=[]
                words=tokens[i].strip().lower().split(' ')    # all words converted into lowercase
                length=len(words)
                total=total+length
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+line
                    exit(0)  
                sent+=[0]*left
                for word in words:
                    #sent.append(word2id.get(word))
                    
                    id=word2id.get(word, -1)   # possibly the embeddings are for words with lowercase
                    if id == -1:    
                        unknown=unknown+1              
                        #sent.append(numpy.random.random_integers(word_count)) 
                        #sent.append(pre_index) # for new words in dev or test data, let's assume its embedding is zero      
                        sent.append(0)           
                    else:
                        sent.append(id)
                        #pre_index=id
                sent+=[0]*right
                data.append(sent)
        senti_file.close()
        print 'Unknown:'+str(unknown)+' rate:'+str(unknown*1.0/total)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
    def load_test_file_preIndex(file, word_count, word2id):
        senti_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        unknown=0
        total=0
        for line in senti_file:
            
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])) # make the label starts from 0 to 4
            for i in range(1,3):
                pre_index=0
                sent=[]
                words=tokens[i].strip().lower().split(' ')    # all words converted into lowercase
                length=len(words)
                total=total+length
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+line
                    exit(0)  
                sent+=[0]*left
                for word in words:
                    #sent.append(word2id.get(word))
                    
                    id=word2id.get(word, -1)   # possibly the embeddings are for words with lowercase
                    if id == -1:       
                        unknown=unknown+1           
                        #sent.append(numpy.random.random_integers(word_count)) 
                        sent.append(pre_index) # for new words in dev or test data, let's assume its embedding is zero                
                    else:
                        sent.append(id)
                        pre_index=id
                sent+=[0]*right
                data.append(sent)
        senti_file.close()
        print 'Unknown:'+str(unknown)+' rate:'+str(unknown*1.0/total)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
    def load_test_file_skipUnknown(file, word_count, word2id):
        senti_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        unknown=0
        total=0
        for line in senti_file:
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])) # make the label starts from 0 to 4
            for i in range(1,3):
                sent=[]
                words=tokens[i].strip().lower().split(' ')    # all words converted into lowercase
                knownWords=[]
                for word in words:
                    id=word2id.get(word, -1) 
                    if id != -1:
                        knownWords.append(word)
                    else:
                        unknown=unknown+1
                
                length=len(knownWords)
                total=total+length
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+line
                    exit(0)  
                sent+=[0]*left
                for word in knownWords:
                    sent.append(word2id.get(word, -1))
                sent+=[0]*right
                data.append(sent)
        senti_file.close()
        print 'Unknown:'+str(unknown)+' rate:'+str(unknown*1.0/total)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
    indices_train, trainY, trainLengths, trainLeftPad, trainRightPad, embeddings, word_count, word2id=load_train_file(trainFile, embeddings, word_count, word2id, train_lines)
    print 'train file loaded over, total pairs:'+str(len(trainLengths)/2)
    if dataMode==1:
        indices_dev, devY, devLengths, devLeftPad, devRightPad=load_dev_file(trainFile, word_count, word2id,train_lines)
        print 'dev file loaded over, total pairs:'+str(len(devLengths)/2)
        indices_test, testY, testLengths, testLeftPad, testRightPad=load_test_file(testFile, word_count, word2id)
        print 'test file loaded over, total pairs:'+str(len(testLengths)/2)
    elif dataMode==2:
        indices_dev, devY, devLengths, devLeftPad, devRightPad=load_dev_file_preIndex(trainFile, word_count, word2id, train_lines)
        print 'dev file loaded over, total pairs:'+str(len(devLengths)/2)
        indices_test, testY, testLengths, testLeftPad, testRightPad=load_test_file_preIndex(testFile, word_count, word2id)   
        print 'test file loaded over, total pairs:'+str(len(testLengths)/2)
    elif dataMode==3:
        indices_dev, devY, devLengths, devLeftPad, devRightPad=load_dev_file_skipUnknown(trainFile, word_count, word2id, train_lines)
        print 'dev file loaded over, total pairs:'+str(len(devLengths)/2)
        indices_test, testY, testLengths, testLeftPad, testRightPad=load_test_file_skipUnknown(testFile, word_count, word2id)   
        print 'test file loaded over, total pairs:'+str(len(testLengths)/2)     

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int32')
        #return shared_y

    embeddings_theano = theano.shared(numpy.asarray(embeddings, dtype=theano.config.floatX), borrow=True)  # @UndefinedVariable
    #embeddings_Q_theano= theano.shared(numpy.asarray(embeddings_Q, dtype=theano.config.floatX), borrow=True)
    train_set_Lengths=shared_dataset(trainLengths)                             
    valid_set_Lengths = shared_dataset(devLengths)
    test_set_Lengths = shared_dataset(testLengths)
    #uni_gram=shared_dataset(unigram)
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    dev_left_pad=shared_dataset(devLeftPad)
    dev_right_pad=shared_dataset(devRightPad)
    test_left_pad=shared_dataset(testLeftPad)
    test_right_pad=shared_dataset(testRightPad)
    
    train_set_y=shared_dataset(trainY)                             
    valid_set_y = shared_dataset(devY)
    test_set_y = shared_dataset(testY)
    

    rval = [(indices_train,train_set_y,train_set_Lengths, train_left_pad, train_right_pad), (indices_dev, valid_set_y, valid_set_Lengths, dev_left_pad, dev_right_pad), (indices_test, test_set_y, test_set_Lengths, test_left_pad, test_right_pad)]
    return rval,      embedding_size, embeddings_theano



def conv_WP(inputs, filters_W, filter_shape, image_shape):
    new_filter_shape=(filter_shape[0], filter_shape[1], 1, filter_shape[3])
    conv_outs=[]

    for i in range(image_shape[2]):
        conv_out_i=conv.conv2d(input=inputs, filters=filters_W[:,:,i:(i+1),:],filter_shape=new_filter_shape, image_shape=image_shape, border_mode='full')
        conv_outs.append(conv_out_i[:,:,i:(i+1),:])
    overall_conv_out=T.concatenate(conv_outs, axis=2)
    
    return overall_conv_out
class Conv_Fold_DynamicK_PoolLayer(object):
    """Pool Layer of a convolutional network """
    
        
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), k=[], unifiedWidth=30, left=[], right=[], firstLayer=True):
        
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
        # the original one
        
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX), borrow=True)
        '''
        self.W = theano.shared(value=numpy.zeros(filter_shape,
                                                 dtype=theano.config.floatX),  # @UndefinedVariable
                                name='W', borrow=True)
        '''
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[2]/2,1), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        bb=T.repeat(self.b, unifiedWidth, axis=1)
        conv_out=  conv_WP(inputs=input, filters_W=self.W, filter_shape=filter_shape, image_shape=image_shape)
        # convolve input feature maps with filters
        #conv_out = conv.conv2d(input=input, filters=self.W,
        #        filter_shape=filter_shape, image_shape=image_shape, border_mode='full')
        #folding
        matrix_shape=T.cast(T.join(0,
                            T.as_tensor([T.prod(conv_out.shape[:-1])]),
                            T.as_tensor([conv_out.shape[3]])),
                            'int64')
        matrix = T.reshape(conv_out, matrix_shape, ndim=2)
        odd_matrix=matrix[0:matrix_shape[0]:2]
        even_matrix=matrix[1:matrix_shape[0]:2]
        raw_folded_matrix=(odd_matrix+even_matrix)*0.5         #here, we should consider average, 
        
        out_shape=T.cast(T.join(0,  conv_out.shape[:-2],
                            T.as_tensor([conv_out.shape[2]/2]),
                            T.as_tensor([conv_out.shape[3]])),
                            'int64')
        fold_out=T.reshape(raw_folded_matrix, out_shape, ndim=4)
        
        self.fold_out=fold_out
        
        padded_matrices=[]
        for i in range(image_shape[0]): # image_shape[0] is actually batch_size
            neighborsForPooling = TSN.images2neibs(ten4=fold_out[i:(i+1)], neib_shape=(1,fold_out.shape[3]), mode='ignore_borders')
            #wenpeng1=theano.printing.Print('original')(neighborsForPooling[:, 25:35])

            non_zeros=neighborsForPooling[:,left[i]:(neighborsForPooling.shape[1]-right[i])] # only consider non-zero elements
            #wenpeng2=theano.printing.Print('non-zeros')(non_zeros)

            neighborsArgSorted = T.argsort(non_zeros, axis=1)
            kNeighborsArg = neighborsArgSorted[:,-k[i]:]
            kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie

            ii = T.repeat(T.arange(non_zeros.shape[0]), k[i])
            jj = kNeighborsArgSorted.flatten()
            pooledkmaxList = non_zeros[ii, jj] # now, should be a vector
            new_shape = T.cast(T.join(0, 
                           T.as_tensor([non_zeros.shape[0]]),
                           T.as_tensor([k[i]])),
                           'int64')
            pooledkmaxMatrix = T.reshape(pooledkmaxList, new_shape, ndim=2)
            if firstLayer:
                leftWidth=(unifiedWidth-k[i])/2
                rightWidth=unifiedWidth-leftWidth-k[i]
                
                left_padding = T.zeros((non_zeros.shape[0], leftWidth), dtype=theano.config.floatX)
                right_padding = T.zeros((non_zeros.shape[0], rightWidth), dtype=theano.config.floatX)
                matrix_padded = T.concatenate([left_padding, pooledkmaxMatrix, right_padding], axis=1) 
                padded_matrices.append(matrix_padded)     
            else:
                padded_matrices.append(pooledkmaxMatrix)
                            
        overall_matrix=T.concatenate(padded_matrices, axis=0)         
        new_shape = T.cast(T.join(0, fold_out.shape[:-2],
                           T.as_tensor([fold_out.shape[2]]),
                           T.as_tensor([unifiedWidth])),
                           'int64')
        pooled_out = T.reshape(overall_matrix, new_shape, ndim=4)
        #wenpeng2=theano.printing.Print('pooled_out')(pooled_out[:,:,:,15:])
        # downsample each feature map individually, using maxpooling
        '''
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)
        '''
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        #@wenpeng:  following tanh operation will voilate our expectation that zero-padding, for its output will have no zero any more
        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        biased_pooled_out=pooled_out + bb.dimshuffle('x', 'x', 0, 1)

        #now, reset some zeros
        self.leftPad=(unifiedWidth-k)/2
        self.rightPad=unifiedWidth-self.leftPad-k
        if firstLayer:
            zero_recover_matrices=[]
            for i in range(image_shape[0]): # image_shape[0] is actually batch_size
                neighborsForPooling = TSN.images2neibs(ten4=biased_pooled_out[i:(i+1)], neib_shape=(1,biased_pooled_out.shape[3]), mode='ignore_borders')     
                left_zeros=T.set_subtensor(neighborsForPooling[:,:self.leftPad[i]], T.zeros((neighborsForPooling.shape[0], self.leftPad[i]), dtype=theano.config.floatX))
                right_zeros=T.set_subtensor(left_zeros[:,(neighborsForPooling.shape[1]-self.rightPad[i]):], T.zeros((neighborsForPooling.shape[0], self.rightPad[i]), dtype=theano.config.floatX))   
                zero_recover_matrices.append(right_zeros)
            overall_matrix_new=T.concatenate(zero_recover_matrices, axis=0)  
            pooled_out_with_zeros = T.reshape(overall_matrix_new, new_shape, ndim=4) 
            self.output=T.tanh(pooled_out_with_zeros)
        else:
            self.output=T.tanh(biased_pooled_out)

        # store parameters of this layer
        self.params = [self.W, self.b]









class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
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
                filter_shape=filter_shape, image_shape=image_shape, border_mode='full')

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

class LeNetConvPoolLayer_valid(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
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
                filter_shape=filter_shape, image_shape=image_shape, border_mode='valid')

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]
if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)