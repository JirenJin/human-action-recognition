require 'hdf5'
require 'torch'
require 'cutorch'
require 'cunn'
require 'optim'
require 'xlua'
require 'paths'
require 'unsup'

-- parse command line arguments
if not opt then
    print('==> processing options')
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text('human activity recognition')
    cmd:text()
    cmd:text('Options:')
    local save_path = 'master_experiments/' .. os.date('%F-%H-%M')
    local worker_path = paths.concat(save_path, 'workers')
    local result_path = paths.concat(save_path, 'results')
    cmd:option('-data', "data.hdf5", 'data path')
    cmd:option('-save', save_path, 'subdirectory to save/log experiments in')
    cmd:option('-worker_path', worker_path, 'subdirectory to save workers in')
    cmd:option('-result_path', result_path, 'subdirectory to save results in')
    cmd:text()
    opt = cmd:parse(arg or {})


    -- create log file
    paths.mkdir(save_path)
    paths.mkdir(worker_path)
    paths.mkdir(result_path)
    cmd:log(paths.concat(opt.save,'log'), opt)
end







local save_path = 'master_experiments/' .. os.date('%F-%H-%M')
local worker_path = paths.concat(save_path, 'workers')
local result_path = paths.concat(save_path, 'results')
local data_path = 'data.hdf5'


local hidden_layer_number_list = {0,1,2,3,4,5}
local hidden_node_number_list = {128,256,512,1024,2048,4096}
local batch_size_list = {1,16,32,64,128,256,512}
local momentum_list = {0.5, 0.9, 0.95, 0.99}
local dropout_list = {0, 0.1, 0.2, 0.3, 0.4, 0.5}
local max_epoch = 100

for iter = 1,100 do
    -- random sample hyperparameters
    local learningRate = torch.pow(10, torch.uniform(-4, 1))
    local hidden_layer_number = hidden_layer_number_list[torch.random(1,#hidden_layer_number_list)]
    local hidden_node_number = hidden_node_number_list[torch.random(1,#hidden_node_number_list)]
    local batch_size = batch_size_list[torch.random(1,#batch_size_list)]
    local momentum = momentum_list[torch.random(1,#momentum_list)]
    local dropout = dropout_list[torch.random(1,#dropout_list)]
    
    os.execute('th worker.lua -save ' .. save_path .. ' -worker_path ' .. worker_path .. ' -result_path ' .. result_path .. ' -data ' .. data_path.. ' -learningRate ' .. learningRate .. ' -hidden_layer_num ' .. hidden_layer_number .. ' -hidden_nodes ' .. hidden_node_number .. ' -batchSize ' .. batch_size .. ' -momentum ' .. momentum .. ' -dropout ' .. dropout .. ' -maxEpoch ' .. max_epoch)
    
end
