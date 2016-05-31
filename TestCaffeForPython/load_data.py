"""
 Specify the path to file d
"""
import os, sys

# Path to your installed caffe 
CAFFE_ROOT = '/media/DATA1/Y/DeepLearningResearch/caffe/'
# Current Path 
CURRENT_DIR = './'
# Path to custom trained model 
CUSTOM_MODEL = ''
# Path to image mean 
IMAGE_MEAN = CAFFE_ROOT + '/data/ilsvrc12/imagenet_mean.binaryproto'
# Path to custom model definition 
MODEL_DEF = ''
# Path to solver  
SOLVER = CAFFE_ROOT + 'examples/cifar10/cifar10_full_solver.prototxt'
# 
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe
# Load trained models for classification 
def load_trained_model(model_number): 
  model_trained = ''
  model_prototxt = ''
  if model_number == 1:
    if os.path.isfile(CAFFE_ROOT + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
        print 'CaffeNet found.'
    else:
        # download if model not already exist
        print 'Downloading pre-trained CaffeNet model...'
        subprocess.call([CAFFE_ROOT + 'scripts/download_model_binary.py', CAFFE_ROOT + 'models/bvlc_reference_caffenet'])

    # Model prototxt file
    model_trained = CAFFE_ROOT + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    # Model caffemodel file
    model_prototxt = CAFFE_ROOT + 'models/bvlc_reference_caffenet/deploy.prototxt'
  elif model_number == 2:
    if os.path.isfile(CAFFE_ROOT + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'):
        print 'AlexNet found.'
    else:
        print 'Downloading pre-trained AlexNet model...'
        subprocess.call([CAFFE_ROOT + 'scripts/download_model_binary.py', CAFFE_ROOT + 'models/bvlc_alexnet'])

    # Model prototxt file
    model_trained = CAFFE_ROOT + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
    # Model caffemodel file
    model_prototxt = CAFFE_ROOT + 'models/bvlc_alexnet/deploy.prototxt'
  elif model_number == 3:
    if os.path.isfile(CAFFE_ROOT + 'models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel'):
        print 'R-CNN found.'
    else:
        print 'Downloading pre-trained R-CNN model...'
        subprocess.call([CAFFE_ROOT + 'scripts/download_model_binary.py', CAFFE_ROOT + 'models/bvlc_reference_rcnn_ilsvrc13'])

    # Model prototxt file
    model_trained = CAFFE_ROOT + 'models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel'
    # Model caffemodel file
    model_prototxt = CAFFE_ROOT + 'models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt'
  else:
    if os.path.isfile(CAFFE_ROOT + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'):
        print 'GoogleLeNet found.'
    else:
        print 'Downloading pre-trained GoogleLeNet model...'
        subprocess.call([CAFFE_ROOT + 'scripts/download_model_binary.py', CAFFE_ROOT + 'models/bvlc_googlenet'])

    # Model prototxt file
    model_trained = CAFFE_ROOT + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
    # Model caffemodel file
    model_prototxt = CAFF_EROOT + 'models/bvlc_googlenet/deploy.prototxt'
  return (model_trained, model_prototxt)

# Load custom models for classification 
def load_custom_model():
  model_trained = CUSTOM_MODEL_FOLDER + 'custom_model.caffemodel'
  model_prototxt = CUSTOM_MODEL_FOLDER + 'custom_deploy.prototxt'
  return (model_trained, model_prototxt)

# Load imagenet labels
def load_imagenet_labels():
  imagenet_labels = CAFFE_ROOT + 'data/ilsvrc12/synset_words.txt'
  with open(imagenet_labels) as f:
    labels = f.readlines()
  return labels
# 
# image for training 
# Get slover for training 
def load_solver():
  return caffe.get_solver(SOLVER)