"""
 Classify input images and extract model features 
"""
import load_data as ld
import os, sys, getopt

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
    with open(imagenet_labels) as f:
        labels = f.readlines()
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
    label_predict(sys.argv[1:])