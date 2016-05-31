import numpy as np
import os, sys, getopt
import subprocess
# Path to your installed caffe 
caffe_root = '/media/yscope/DATA1/Y/DeepLearningResearch/caffe/'

# a switch loop for choose trained model 
# 1 : caffe net 
# 2: alexnet 
# 3 : R-CNN
# 4: AlexNet
choose = 0
print("1: caffe net ")
print("2: alexnet  ")
print("3: R-CNN ")
print("4: GoogleLeNet ")
choose = input("choose trained model: ")
while (choose < 1 or choose > 5):
    print("1: caffe net ")
    print("2: alexnet  ")
    print("3: R-CNN ")
    print("4: AlexNet ")
    choose = input("choose trained model: ")

if choose == 1:
    if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
        print 'CaffeNet found.'
    else:
        # download if model not already exist
        print 'Downloading pre-trained CaffeNet model...'
        subprocess.call([caffe_root + 'scripts/download_model_binary.py', caffe_root + 'models/bvlc_reference_caffenet'])

    # Model prototxt file
    model_trained = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    # Model caffemodel file
    model_prototxt = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
elif choose == 2:
    if os.path.isfile(caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'):
        print 'AlexNet found.'
    else:
        print 'Downloading pre-trained AlexNet model...'
        subprocess.call([caffe_root + 'scripts/download_model_binary.py', caffe_root + 'models/bvlc_alexnet'])

    # Model prototxt file
    model_trained = caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'
    # Model caffemodel file
    model_prototxt = caffe_root + 'models/bvlc_alexnet/deploy.prototxt'
elif choose == 3:
    if os.path.isfile(caffe_root + 'models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel'):
        print 'R-CNN found.'
    else:
        print 'Downloading pre-trained R-CNN model...'
        subprocess.call([caffe_root + 'scripts/download_model_binary.py', caffe_root + 'models/bvlc_reference_rcnn_ilsvrc13'])

    # Model prototxt file
    model_trained = caffe_root + 'models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel'
    # Model caffemodel file
    model_prototxt = caffe_root + 'models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt'
else:
    if os.path.isfile(caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'):
        print 'GoogleLeNet found.'
    else:
        print 'Downloading pre-trained GoogleLeNet model...'
        subprocess.call([caffe_root + 'scripts/download_model_binary.py', caffe_root + 'models/bvlc_googlenet'])

    # Model prototxt file
    model_trained = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
    # Model caffemodel file
    model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'

# File containing the class labels
imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
sys.path.insert(0, caffe_root + 'python')
import caffe
def label_predict(argv):
    try:
            opts, args = getopt.getopt(argv,"hi:o:",["ifolder="])
    except getopt.GetoptError:
            print 'Test_python.py -i <inputfolder> '
            sys.exit(2)
    for opt, arg in opts:
            if opt == '-h':
                print 'Test_python.py -i <inputfolder>'
                sys.exit()
            elif opt in ("-i"):
                inputfolder = arg

    caffe.set_device(0)
    caffe.set_mode_gpu()

    # declare net, use  test mode ( don't perform dropout )
    net = caffe.Net(model_prototxt, model_trained, caffe.TEST) 
    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    # set the size of the input
    net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227
    # Loading class labels
    with open(imagenet_labels) as f:
        labels = f.readlines()
    for image in os.listdir(inputfolder):
        # load image to data layer 
        im = caffe.io.load_image(inputfolder + image)
        net.blobs['data'].data[...] = transformer.preprocess('data', im)
        # compute 
        output = net.forward()
        output_prob = output['prob'][0] #the output probability vector for the first image in the batch
        # print predicted label 
        print 'output label:', labels[output_prob.argmax()]
if __name__ == "__main__":
    label_predict(sys.argv[1:])
