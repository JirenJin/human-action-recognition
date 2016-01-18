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
    local save_path = 'worker_experiments/' .. os.date('%F-%H-%M')
    local model_path = paths.concat(save_path, 'models')
    local data_path = 'data.hdf5'
    cmd:option('-data', data_path, 'data path')
    cmd:option('-save', save_path, 'subdirectory to save/log experiments in')
    cmd:option('-model_save', model_path, 'subdirectory to save models in')
    cmd:option('-plot', false, 'live plot')
    cmd:option('-coefL1', 0.001, 'L1 penalty on the weights')
    cmd:option('-coefL2', 0.001, 'L2 penalty on the weights')
    cmd:option('-hidden_nodes', 1024, 'number of hidden nodes in each layer')
    cmd:option('-hidden_layer_num', 1, 'number of hidden layers')
    cmd:option('-early_stop', 10, 'early-stopping number')
    cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
    cmd:option('-learningRateDecay', 0, 'learning rate Decay')
    cmd:option('-batchSize', 128, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
    cmd:option('-momentum', 0.9, 'momentum (SGD only)')
    cmd:option('-dropout', 0, 'dropout')
    cmd:option('-maxEpoch', 10, 'max number of epoch')
    cmd:text()
    opt = cmd:parse(arg or {})


    -- create log file
    paths.mkdir(opt.save)
    paths.mkdir(opt.model_save)
    cmd:log(paths.concat(opt.save,'log'), opt)
end


-- log results to files
local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local valLogger = optim.Logger(paths.concat(opt.save, 'val.log'))

torch.manualSeed(1234)

-- define target classes
classes = {'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING',
'STANDING', 'LAYING'}

-- load training and test data
local myFile = hdf5.open(opt.data, 'r')
local train_x = myFile:read('train_x'):all()
local train_y = myFile:read('train_y'):all():cuda()
local val_x = myFile:read('val_x'):all()
local val_y = myFile:read('val_y'):all():cuda()
local test_x = myFile:read('test_x'):all()
-- zca whitening for train data, 
-- also store mean, and p matrix for validation and test data transformation
train_x, zca_mean, zca_p, zca_invp = unsup.zca_whiten(train_x)
val_x = unsup.zca_whiten(val_x, zca_mean, zca_p, zca_invp)
test_x = unsup.zca_whiten(test_x, zca_mean, zca_p, zca_invp)
train_x = train_x:cuda()
val_x = val_x:cuda()
test_x = test_x:cuda()
myFile:close()
collectgarbage()

print('train set size: ' .. train_x:size(1))
print('validation set size: ' .. val_x:size(1))
print('test set size: ' .. test_x:size(1))

-- define deep neural networks
local hidden_nodes = opt.hidden_nodes
local model = nn.Sequential()
if opt.hidden_layer_num > 0 then
    model:add(nn.Linear(train_x:size(2), hidden_nodes))
    model:add(nn.ReLU())
    model:add(nn.Dropout(opt.dropout))
    for i = 1, opt.hidden_layer_num - 1 do
        model:add(nn.Linear(hidden_nodes, hidden_nodes))
        model:add(nn.ReLU())
        model:add(nn.Dropout(opt.dropout))
    end
    model:add(nn.Linear(hidden_nodes, #classes))
else
    -- no hidden layer: linear
    model:add(nn.Linear(train_x:size(2), #classes))
end
model:add(nn.LogSoftMax())
model = model:cuda()
local criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()
local w, dw = model:getParameters()

-- verbose
print('<human activity recognition> using model:')
print(model)

-- set variables for early-stopping
local train_e = {}
local val_e = {}
local train_acc = {}
local val_acc = {}
local overfit_count = {}

-- hyperparameters for learning
local learning_params = {
   learningRate = opt.learningRate,
   learningRateDecay = opt.learningRateDecay,
   weightDecay = opt.weightDecay,
   momentum = opt.momentum
}

-- this matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

function train(epoch)
    collectgarbage()
    -- set model to training mode
    model:training()
    -- epoch training start time
    local time = sys.clock()
    -- shuffle at each epoch
    shuffle = torch.randperm(train_x:size(1))
    -- do one epoch
    print('\n<trainer> on training set:')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    local total_loss = 0
    for t = 1,train_x:size(1),opt.batchSize do
        -- create mini batch
        local inputs = torch.CudaTensor(opt.batchSize, train_x:size(2))
        local targets = torch.CudaTensor(opt.batchSize)
        local k = 1
        for i = t,math.min(t+opt.batchSize-1,train_x:size(1)) do
            -- load new sample
            local input = train_x[shuffle[i]]
            local target = train_y[shuffle[i]]
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end
        -- define evaluate function for optim
        local function feval(new_w)
            
            if w~=new_w then
                w:copy(new_w)
            end
     
            -- reset gradients
            dw:zero()
            
            -- iterately choose training sample
            local predictions = model:forward(inputs)
            local loss = criterion:forward(predictions, targets)
            total_loss = total_loss + loss * inputs:size(1)
            model:backward(inputs, criterion:backward(predictions, targets))
            
            -- regularization
            if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
                -- locals:
                local norm,sign= torch.norm,torch.sign

                -- Loss:
                loss = loss + opt.coefL1 * norm(w,1)
                loss = loss + opt.coefL2 * norm(w,2)^2/2

                -- Gradients:
                dw:add( sign(w):mul(opt.coefL1) + w:clone():mul(opt.coefL2) )
            end
            
            -- important to avoid errors
            collectgarbage()
            -- update confusion
            for j = 1,predictions:size(1) do
                confusion:add(predictions[j], targets[j])
            end
            
            return loss, dw
        end
        
        optim.sgd(feval, w, learning_params)
    end
    
    collectgarbage()
    
    -- time taken
    time = sys.clock() - time
    time = time / train_x:size(1)
    print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')
    local train_err = total_loss/train_x:size(1)
    print(confusion)
    local train_accuracy = confusion.totalValid * 100
    confusion:zero()
    --print('train_err: ' .. train_err)
    print('train_accuracy: ' .. train_accuracy .. "%")
    trainLogger:add{['% mean accuracy (train set)'] = train_accuracy}
    if opt.plot then
        trainLogger:style{['% mean accuracy (train set)'] = '-'}
        trainLogger:plot()
    end
    
    return train_err, train_accuracy
end

function val()
    collectgarbage()
    model:evaluate()
    local current_loss = 0
    
    for t = 1,val_x:size(1),opt.batchSize do
        -- create mini batch for test set
        local inputs = torch.CudaTensor(opt.batchSize, val_x:size(2))
        local targets = torch.CudaTensor(opt.batchSize)
        local k = 1
        for i = t,math.min(t+opt.batchSize-1,val_x:size(1)) do
            -- load new sample
            local input = val_x[i]
            local target = val_y[i]
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end
        preds = model:forward(inputs)
        loss = criterion:forward(preds, targets)
        current_loss = current_loss + loss * opt.batchSize
        for j = 1,preds:size(1) do
            confusion:add(preds[j], targets[j])
        end
    end
    
    val_err = current_loss/val_x:size(1)
    print(confusion)
    val_accuracy = confusion.totalValid * 100
    -- reset confusion matrix
    confusion:zero()
    print('val_accuracy: ' .. val_accuracy .. '%')
    --update log/plot
    valLogger:add{['% mean accuracy (validation set)'] = val_accuracy}
    if opt.plot then
        valLogger:style{['% mean accuracy (validation set)'] = '-'}
        valLogger:plot()
    end
    return val_err, val_accuracy
end

function test()
    model:evaluate()
    local f = torch.DiskFile(opt.save .. '/result.txt', 'w')
    for i=1, test_x:size(1) do
        local nll, pred = torch.max(model:forward(test_x[i]):float(), 1)
        f:writeFloat(pred[1])
    end
    f:close()
end

local best_val = nil
local best_model = nil
for epoch = 1,opt.maxEpoch do
    -- training
    model:training()
    train_e[epoch], train_acc[epoch] = train(epoch)
    -- validation
    model:evaluate()
    val_e[epoch], val_acc[epoch] = val()
    -- initialize the best model on val
    if epoch == 1 then
        best_val = val_acc[1]
        best_model = model:clone()
    end
    if epoch > 1 then
        --[[if val_e[epoch] > val_e[epoch - 1] then
            learning_params['learningRate'] = learning_params['learningRate'] * 0.95
            print('Learning rate changes to '.. learning_params['learningRate'])
        end--]]
        if val_acc[epoch] > best_val then
            best_val = val_acc[epoch]
            best_model = model:clone()
        end
    end
    
    --[[
    -- early-stopping: if val_e increases when train_e decreases for continued 5 times, stop training
    overfit_count[epoch] = 0
    if epoch > opt.early_stop then
        --if (train_e[epoch] < train_e[epoch - 1]) and (val_e[epoch] > val_e[epoch - 1]) then
        if (val_e[epoch] > val_e[epoch - 1]) then
            overfit_count[epoch] = 1
            count = 0
            for j=0,opt.early_stop-1 do
                count = count + overfit_count[epoch - j]
            end
            if count == opt.early_stop then 
                break
            end
        end
    end
    --]]
end

local best_val_acc, best_val_model = torch.max(torch.Tensor(val_acc),1)
print('maximus validation accuracy: ' .. best_val_acc[1] .. '%')
print('best model on validation set: ' .. best_val_model[1])
local best_train_acc, best_train_model = torch.max(torch.Tensor(train_acc), 1)
print('maximum train accuracy: ' .. best_train_acc[1] .. '%')
print('best model on train set: ' .. best_train_model[1])


-- create worker checkpoint
local worker = {}
worker['model'] = best_model
worker['train_acc'] = train_acc
worker['val_acc'] = val_acc
worker['options'] = opt
torch.save(opt.save .. '/worker-' .. best_val .. '.t7', worker)
-- save the best model on val
-- torch.save(opt.model_save .. '/model-' .. best_val .. '.t7', best_model)


-- choose the best model for validation to test
--local model = torch.load(opt.model_save .. '/model-' .. best_val_model[1] .. '.t7')
model = best_model
model:evaluate()
test()

