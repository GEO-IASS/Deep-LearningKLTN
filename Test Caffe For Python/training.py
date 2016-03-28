"""
  train networks and store result 
""" 
import load_data as ld

# load solver from file
solver = caffe.get_solver(ld.SOLVER)
# run solver to train net
solver.solve() 
# compute accuracy of the model
accuracy = 0
batch_size = solver.test_nets[0].blobs['data'].num
# compute test_iters 
for i in range(test_iters):
    solver.test_nets[0].forward()
    accuracy += solver.test_nets[0].blobs['accuracy'].data
accuracy /= test_iters

print("Accuracy: {:.3f}".format(accuracy))
# save trained model to file 
