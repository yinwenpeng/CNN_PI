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
from mlp import HiddenLayer

def read_data_HK(trainFile, devFile, testFile, emb_file, maxlength, useEmb):
    #first store emb_file into a dict
    embeddings=[]
    
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
            embeddings.append(vector)
            word2id[tokens[0]]=num_lines+1 # word index starts from 1
    
        embeddingsFile.close()
    else:
        embedding_size=48
        embeddings.append(numpy.zeros(embedding_size)) 
    word_count=len(embeddings)
    print 'Totally, '+str(word_count)+' word embeddings.'
    
    def load_train_file(file, embeddings, word_count, word2id):   
        senti_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        #leftPad=[]
        for line in senti_file:
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])-1) # make the label starts from 0 to 4
            sent=[]
            words=tokens[1].split(' ')
            length=len(words)
            Lengths.append(length)
            #left=(maxlength-length)/2
            right=maxlength-length
            #leftPad.append(left)
            
            if right<0:
                print 'Too long sentence:\n'+line
                exit(0)   
            for word in words:
                #sent.append(word2id.get(word))
                
                id=word2id.get(word, -1)   # possibly the embeddings are for words with lowercase
                if id == -1:
                    embeddings.append(numpy.random.uniform(-1,1,embedding_size)) # generate a random embedding for an unknown word
                    word2id[word]=word_count
                    sent.append(word_count)
                    word_count=word_count+1                  
                else:
                    sent.append(id)
            for i in range(right):
                sent.append(0)
            data.append(sent)
        senti_file.close()
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(embeddings), word_count, word2id
    def load_dev_or_test_file(file, word_count, word2id):
        senti_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        for line in senti_file:
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])-1) # make the label starts from 0 to 4
            sent=[]
            words=tokens[1].split(' ')
            length=len(words)
            Lengths.append(length)
            right=maxlength-length
            if right<0:
                print 'Too long sentence:\n'+line
                exit(0)  
            for word in words:
                #sent.append(word2id.get(word))
                
                id=word2id.get(word, -1)   # possibly the embeddings are for words with lowercase
                if id == -1:                  
                    #sent.append(numpy.random.random_integers(word_count)) 
                    sent.append(0) # for new words in dev or test data, let's assume its embedding is zero                 
                else:
                    sent.append(id)
            for i in range(right):
                sent.append(0)
            data.append(sent)
        senti_file.close()
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths)    
    indices_train, trainY, trainLengths, embeddings, word_count, word2id=load_train_file(trainFile, embeddings, word_count, word2id)
    print 'train file loaded over, totally:'+str(len(trainLengths))
    indices_dev, devY, devLengths=load_dev_or_test_file(devFile, word_count, word2id)
    print 'dev file loaded over, totally:'+str(len(devLengths))
    indices_test, testY, testLengths=load_dev_or_test_file(testFile, word_count, word2id)
    print 'test file loaded over, totally:'+str(len(testLengths))

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int32')
        #return shared_y

    embeddings_theano = theano.shared(numpy.asarray(embeddings, dtype=theano.config.floatX), borrow=True)  # @UndefinedVariable

    train_set_Lengths=shared_dataset(trainLengths)                             
    valid_set_Lengths = shared_dataset(devLengths)
    test_set_Lengths = shared_dataset(testLengths)
    
    train_set_y=shared_dataset(trainY)                             
    valid_set_y = shared_dataset(devY)
    test_set_y = shared_dataset(testY)
    

    rval = [(indices_train,train_set_y,train_set_Lengths), (indices_dev, valid_set_y, valid_set_Lengths), (indices_test, test_set_y, test_set_Lengths)]
    return rval,      embedding_size, embeddings_theano
class ConvFoldPoolLayer(object):
    """Pool Layer of a convolutional network """
    def kmaxPooling(self, fold_out, k):
        neighborsForPooling = TSN.images2neibs(ten4=fold_out, neib_shape=(1,fold_out.shape[3]), mode='ignore_borders')
        self.neighbors = neighborsForPooling

        neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-k:]
        #self.bestK = kNeighborsArg
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1)

        ii = T.repeat(T.arange(neighborsForPooling.shape[0]), k)
        jj = kNeighborsArgSorted.flatten()
        pooledkmaxTmp = neighborsForPooling[ii, jj]
        new_shape = T.cast(T.join(0, fold_out.shape[:-2],
                           T.as_tensor([fold_out.shape[2]]),
                           T.as_tensor([k])),
                           'int64')
        pooled_out = T.reshape(pooledkmaxTmp, new_shape, ndim=4)  
                
        return pooled_out
    def folding(self, curConv_out):
        #folding
        matrix_shape=T.cast(T.join(0,
                            T.as_tensor([T.prod(curConv_out.shape[:-1])]),
                            T.as_tensor([curConv_out.shape[3]])),
                            'int64')
        matrix = T.reshape(curConv_out, matrix_shape, ndim=2)
        odd_matrix=matrix[0:matrix_shape[0]:2]
        even_matrix=matrix[1:matrix_shape[0]:2]
        raw_folded_matrix=odd_matrix+even_matrix
        
        out_shape=T.cast(T.join(0,  curConv_out.shape[:-2],
                            T.as_tensor([curConv_out.shape[2]/2]),
                            T.as_tensor([curConv_out.shape[3]])),
                            'int64')
        fold_out=T.reshape(raw_folded_matrix, out_shape, ndim=4)
        return fold_out
    def conv_folding_Pool(self, bInd):
        curInput = self.input[bInd:bInd+1, :, :, :] #each sentence
        lengthForConv = self.dynamicK[bInd]
        inputForConv = curInput[:,:,:,0:lengthForConv]
        curConv_out = conv.conv2d(input=inputForConv, filters=self.W,
              filter_shape=self.filter_shape, image_shape=None, border_mode='full') # full means wide convolution
        k = self.k
        fold_out=self.folding(curConv_out)
        
        return self.kmaxPooling(fold_out, self.k)

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), k=4, dynamicK=[]):
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
        self.k=k
        self.dynamicK=dynamicK
        self.filter_shape=filter_shape
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

        bInd = 0
        pooled_out = self.conv_folding_Pool(bInd)
        for bInd in range(1, image_shape[0]):
            pooled_out = T.concatenate([pooled_out, self.conv_folding_Pool(bInd)], axis = 0)




        '''
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)
        
        #folding
        matrix_shape=T.cast(T.join(0,
                            T.as_tensor([T.prod(conv_out.shape[:-1])]),
                            T.as_tensor([conv_out.shape[3]])),
                            'int64')
        matrix = T.reshape(conv_out, matrix_shape, ndim=2)
        odd_matrix=matrix[0:matrix_shape[0]:2]
        even_matrix=matrix[1:matrix_shape[0]:2]
        raw_folded_matrix=odd_matrix+even_matrix
        
        out_shape=T.cast(T.join(0,  conv_out.shape[:-2],
                            T.as_tensor([conv_out.shape[2]/2]),
                            T.as_tensor([conv_out.shape[3]])),
                            'int64')
        fold_out=T.reshape(raw_folded_matrix, out_shape, ndim=4)
        
        matrices=[]
        for i in range(image_shape[0]): # image_shape[0] is actually batch_size
            neighborsForPooling = TSN.images2neibs(ten4=fold_out[i:(i+1)], neib_shape=(1,fold_out.shape[3]), mode='ignore_borders')
            non_zeros=neighborsForPooling[:,left[i]:(neighborsForPooling.shape[1]-right[i])]
            #neighborsForPooling=neighborsForPooling[:,leftBound:(rightBound+1)] # only consider non-zero elements
            
            neighborsArgSorted = T.argsort(non_zeros, axis=1)
            kNeighborsArg = neighborsArgSorted[:,-k:]
            kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie

            ii = T.repeat(T.arange(non_zeros.shape[0]), k)
            jj = kNeighborsArgSorted.flatten()
            pooledkmaxTmp = non_zeros[ii, jj] # now, should be a vector
            new_shape = T.cast(T.join(0, 
                           T.as_tensor([non_zeros.shape[0]]),
                           T.as_tensor([k])),
                           'int64')
            pooledkmaxTmp = T.reshape(pooledkmaxTmp, new_shape, ndim=2)
            matrices.append(pooledkmaxTmp)
        
        overall_matrix=T.concatenate(matrices, axis=0)         
        new_shape = T.cast(T.join(0, fold_out.shape[:-2],
                           T.as_tensor([fold_out.shape[2]]),
                           T.as_tensor([k])),
                           'int64')
        pooled_out = T.reshape(overall_matrix, new_shape, ndim=4)      
        '''
        
        
        
        
        
        
        
        '''
        #k-max, but without getting ride of zero on both sides
        neighborsForPooling = TSN.images2neibs(ten4=fold_out, neib_shape=(1,fold_out.shape[3]), mode='ignore_borders')

        #k = poolsize[1]

        neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-k:]
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1)

        ii = T.repeat(T.arange(neighborsForPooling.shape[0]), k)
        jj = kNeighborsArgSorted.flatten()
        pooledkmaxTmp = neighborsForPooling[ii, jj]

        # reshape pooledkmaxTmp
        new_shape = T.cast(T.join(0, fold_out.shape[:-2],
                           T.as_tensor([fold_out.shape[2]]),
                           T.as_tensor([k])),
                           'int64')
        pooled_out = T.reshape(pooledkmaxTmp, new_shape, ndim=4)
        '''
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
        
        
        
class Conv_DynamicK_PoolLayer(object):
    """Pool Layer of a convolutional network """
    def dynamic_kmaxPooling(self, curConv_out, k):
        neighborsForPooling = TSN.images2neibs(ten4=curConv_out, neib_shape=(1,curConv_out.shape[3]), mode='ignore_borders')
        self.neighbors = neighborsForPooling

        neighborsArgSorted = T.argsort(neighborsForPooling, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-k:]
        #self.bestK = kNeighborsArg
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1)

        ii = T.repeat(T.arange(neighborsForPooling.shape[0]), k)
        jj = kNeighborsArgSorted.flatten()
        pooledkmaxTmp = neighborsForPooling[ii, jj]
        new_shape = T.cast(T.join(0, 
                           T.as_tensor([neighborsForPooling.shape[0]]),
                           T.as_tensor([k])),
                           'int64')
        pooledkmax_matrix = T.reshape(pooledkmaxTmp, new_shape, ndim=2)

        rightWidth=self.unifiedWidth-k            
        right_padding = T.zeros((neighborsForPooling.shape[0], rightWidth), dtype=theano.config.floatX)
        matrix_padded = T.concatenate([pooledkmax_matrix, right_padding], axis=1)      
        #recover tensor form
        new_shape = T.cast(T.join(0, curConv_out.shape[:-2],
                           T.as_tensor([curConv_out.shape[2]]),
                           T.as_tensor([self.unifiedWidth])),
                           'int64')

        curPooled_out = T.reshape(matrix_padded, new_shape, ndim=4)
                
        return curPooled_out
    def folding(self, curConv_out):
        #folding
        matrix_shape=T.cast(T.join(0,
                            T.as_tensor([T.prod(curConv_out.shape[:-1])]),
                            T.as_tensor([curConv_out.shape[3]])),
                            'int64')
        matrix = T.reshape(curConv_out, matrix_shape, ndim=2)
        odd_matrix=matrix[0:matrix_shape[0]:2]
        even_matrix=matrix[1:matrix_shape[0]:2]
        raw_folded_matrix=odd_matrix+even_matrix
        
        out_shape=T.cast(T.join(0,  curConv_out.shape[:-2],
                            T.as_tensor([curConv_out.shape[2]/2]),
                            T.as_tensor([curConv_out.shape[3]])),
                            'int64')
        fold_out=T.reshape(raw_folded_matrix, out_shape, ndim=4)
        return fold_out
    def convAndPoolStep(self, bInd):
        curInput = self.input[bInd:bInd+1, :, :, :] #each sentence
        lengthForConv = self.sentLengths[bInd]
        inputForConv = curInput[:,:,:,0:lengthForConv]
        curConv_out = conv.conv2d(input=inputForConv, filters=self.W,
              filter_shape=self.filter_shape, image_shape=None, border_mode='full') # full means wide convolution
        fold_out=self.folding(curConv_out)
        k = self.k[bInd]
        return self.dynamic_kmaxPooling(fold_out, k)

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), sentLengths=[], k=[], unifiedWidth=20):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.sentLengths=sentLengths
        self.k=k
        self.unifiedWidth=unifiedWidth
        self.filter_shape=filter_shape
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
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)



        bInd = 0
        pooled_out = self.convAndPoolStep(bInd)
        for bInd in range(1, image_shape[0]):
            pooled_out = T.concatenate([pooled_out, self.convAndPoolStep(bInd)], axis = 0)
        
        
        
        '''
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)
        #conv_out_print=theano.printing.Print('conv_out')(conv_out[:,:,:,25:35])
        padded_matrices=[]
        #leftPad=[]
        #rightPad=[]
        for i in range(image_shape[0]): # image_shape[0] is actually batch_size
            neighborsForPooling = TSN.images2neibs(ten4=conv_out[i:(i+1)], neib_shape=(1,conv_out.shape[3]), mode='ignore_borders')
            #wenpeng1=theano.printing.Print('original')(neighborsForPooling[:, 25:35])

            non_zeros=neighborsForPooling[:,left[i]:(neighborsForPooling.shape[1]-right[i])] # only consider non-zero elements
            #wenpeng2=theano.printing.Print('non-zeros')(wocao)

            neighborsArgSorted = T.argsort(non_zeros, axis=1)
            kNeighborsArg = neighborsArgSorted[:,-k[i]:]
            kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie

            ii = T.repeat(T.arange(non_zeros.shape[0]), k[i])
            jj = kNeighborsArgSorted.flatten()
            pooledkmaxTmp = non_zeros[ii, jj] # now, should be a vector
            new_shape = T.cast(T.join(0, 
                           T.as_tensor([non_zeros.shape[0]]),
                           T.as_tensor([k[i]])),
                           'int64')
            pooledkmaxTmp = T.reshape(pooledkmaxTmp, new_shape, ndim=2)
            
            leftWidth=(unifiedWidth-k[i])/2
            rightWidth=unifiedWidth-leftWidth-k[i]

            #leftPad.append(leftWidth)
            #rightPad.append(rightWidth)
            
            left_padding = T.zeros((non_zeros.shape[0], leftWidth), dtype=theano.config.floatX)
            right_padding = T.zeros((non_zeros.shape[0], rightWidth), dtype=theano.config.floatX)
            matrix_padded = T.concatenate([left_padding, pooledkmaxTmp, right_padding], axis=1) 
            padded_matrices.append(matrix_padded)     
                            
        overall_matrix=T.concatenate(padded_matrices, axis=0)         
        new_shape = T.cast(T.join(0, conv_out.shape[:-2],
                           T.as_tensor([conv_out.shape[2]]),
                           T.as_tensor([unifiedWidth])),
                           'int64')
        pooled_out = T.reshape(overall_matrix, new_shape, ndim=4)
        '''
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
        biased_pooled_out=pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        #now, reset some zeros
        self.rightPad=self.unifiedWidth-k
        '''
        #actually, need not recover zeros
        zero_recover_matrices=[]
        for i in range(image_shape[0]): # image_shape[0] is actually batch_size
            neighborsForPooling = TSN.images2neibs(ten4=biased_pooled_out[i:(i+1)], neib_shape=(1,biased_pooled_out.shape[3]), mode='ignore_borders')     
            #left_zeros=T.set_subtensor(neighborsForPooling[:,:self.leftPad[i]], T.zeros((neighborsForPooling.shape[0], self.leftPad[i]), dtype=theano.config.floatX))
            right_zeros=T.set_subtensor(neighborsForPooling[:,-self.rightPad[i]:], T.zeros((neighborsForPooling.shape[0], self.rightPad[i]), dtype=theano.config.floatX))   
            zero_recover_matrices.append(right_zeros)
        overall_matrix_new=T.concatenate(zero_recover_matrices, axis=0)  
        new_shape = T.cast(T.join(0, pooled_out.shape[:-2],
                           T.as_tensor([pooled_out.shape[2]]),
                           T.as_tensor([self.unifiedWidth])),
                           'int64')
        pooled_out_with_zeros = T.reshape(overall_matrix_new, new_shape, ndim=4)        
        self.output=T.tanh(pooled_out_with_zeros)
        '''
        self.output=T.tanh(biased_pooled_out)
        # store parameters of this layer
        self.params = [self.W, self.b]

        
def dropout_from_layer(rng,layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output



def shared_dataset(data_y, borrow=True):
    shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
    return T.cast(shared_y, 'int32')