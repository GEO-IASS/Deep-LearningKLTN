"""
  Read CIFAR-100 dataset and convert to LMDB database
"""
import cPickle 
import sys
import numpy 
import inspect 
# path to cifar training data 
TRAIN = "/media/yscope/DATA1/Y/DeepLearningResearch/cifar-100-python/train"
def main(argv):
  fo = open(TRAIN, 'rb')
  dict = cPickle.load(fo)
  fo.close()
  print dict.itervalues().next()
if __name__ == "__main__":
  main(sys.argv)