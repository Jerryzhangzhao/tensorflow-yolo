import sys
from optparse import OptionParser

sys.path.append('./')

import yolo
from yolo.utils.process_config import process_config

parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure",  
                  help="configure filename")
(options, args) = parser.parse_args() 
if options.configure:
  conf_file = str(options.configure)
else:
  print('please sspecify --conf configure filename')
  exit(0)

common_params, dataset_params, net_params, solver_params = process_config(conf_file)

#yolo.dataset.text_dataset.TextDataSet ==>data process
dataset = eval(dataset_params['name'])(common_params, dataset_params)

#yolo.net.yolo_tiny_net.YoloTinyNet ==>network,loss
net = eval(net_params['name'])(common_params, net_params)

#yolo.solver.yolo_solver.YoloSolver
solver = eval(solver_params['name'])(dataset, net, common_params, solver_params)

solver.solve()
