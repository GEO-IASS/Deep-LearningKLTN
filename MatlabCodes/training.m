
if exist('./matlab/+caffe', 'dir')
  addpath('./matlab');
else
  error('Please copy matlab folder from caffe folder to parent folder !');
end
% load model for training 
MODEL = '../caffe/examples/cifar10/cifar10_full.prototxt';
% load solver 
SOLVER = '../caffe/examples/cifar10/cifar10_full_solver.prototxt';
% load solver 
solver = caffe.Solver(SOLVER);
% run train
solver.solve()