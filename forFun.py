from theano import  config
import numpy
print config.floatX   # either 'float32' or 'float64'
x = numpy.asarray([1,2,3], dtype=config.floatX)

print x