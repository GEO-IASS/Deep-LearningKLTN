"""
 Classify input images and extract model features 
"""
import numpy as np
import load_data as ld
import os, sys, getopt
import caffe 

def label_predict(argv):
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifolder="])
    except getopt.GetoptError:
        print 'classifier.py -i <inputfolder> '
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'classifier.py -i <inputfolder>'
            sys.exit()
        elif opt in ("-i"):
            inputfolder = arg
    # choose model for classificaiton 
    choose = 0
    print("1: caffe net ")
    print("2: alexnet  ")
    print("3: R-CNN ")
    print("4: GoogleLeNet ")
    print("5: Custom_model ")
    choose = input("choose trained model: ")
    while (choose < 1 or choose > 5):
        print("1: caffe net ")
        print("2: alexnet  ")
        print("3: R-CNN ")
        print("4: AlexNet ")
        choose = input("choose trained model: ")
    model_trained, model_prototxt = ld.load_trained_model(choose)
    # run in gpu mode 
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # declare net, use  test mode ( don't perform dropout )
    net = caffe.Net(model_prototxt, model_trained, caffe.TEST) 
    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(ld.CAFFE_ROOT + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    # set the size of the input
    net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227
    # Loading class labels
    labels = ld.load_imagenet_labels()
    for image in os.listdir(inputfolder):
        # load image to data layer 
        im = caffe.io.load_image(inputfolder + image)
        net.blobs['data'].data[...] = transformer.preprocess('data', im)
        # compute 
        output = net.forward()
        # Show layer's parameters and activation  
        for layer_name, param in net.params.iteritems():
          print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
        output_prob = output['prob'][0] #the output probability vector for the first image in the batch
        # print predicted label 
        print 'output label:', labels[output_prob.argmax()]
        # print some other features 
if __name__ == "__main__":
    label_predict(sys.argv)