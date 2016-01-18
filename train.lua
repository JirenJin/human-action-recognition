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
    local save_path = 'experiments/' .. os.date('%F-%H-%M')
    local model_path = paths.concat(save_path, 'models')
    local data_path = 'data.hdf5'
    cmd:option('-data', data_path, 'data path')
    cmd:option('-save', save_path, 'subdirectory to save/log experiments in')
    cmd:option('-model_save', model_path, 'subdirectory to save models in')
    cmd:option('-plot', false, 'live plot')
    cmd:option('-coefL1', 0, 'L1 penalty on the weights')
    cmd:option('-coefL2', 0, 'L2 penalty on the weights')
    cmd:option('-hidden_nodes', 1024, 'number of hidden nodes in each layer')
    cmd:option('-hidden_layer_num', 1, 'number of hidden layers')
    cmd:option('-early_stop', 10, 'early-stopping number')
    cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
    cmd:option('-learningRateDecay', 0, 'learning rate Decay')
    cmd:option('-batchSize', 128, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
    cmd:option('-momentum', 0.9, 'momentum (SGD only)')
    cmd:option('-dropout', 0.5, 'dropout')
    cmd:text()
    opt = cmd:parse(arg or {})


    -- create log file
    paths.mkdir(opt.save)
    paths.mkdir(opt.model_save)
    cmd:log(paths.concat(opt.save,'log'), opt)
end

-- copy train.lua to experiments directory
os.execute('cp train.lua ' .. opt.save)

-- log results to files
local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local valLogger = optim.Logger(paths.concat(opt.save, 'val.log'))

torch.manualSeed(1234)

classes = {'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING',
'STANDING', 'LAYING'}

local myFile = hdf5.open(opt.data, 'r')
local train_x = myFile:read('train_x'):all()

train_mean = train_x:mean(1)
train_std = train_x:std(1)
-- zca whitening for train data, 
-- also store mean, and p matrix for validation and test data transformation
train_x, zca_mean, zca_p, zca_invp = unsup.zca_whiten(train_x)
local train_y = myFile:read('train_y'):all():cuda()
local val_x = myFile:read('val_x'):all()
val_x = unsup.zca_whiten(val_x, zca_mean, zca_p, zca_invp)
val_x = val_x:cuda()
local val_y = myFile:read('val_y'):all():cuda()
local test_x = myFile:read('test_x'):all()
test_x = unsup.zca_whiten(test_x, zca_mean, zca_p, zca_invp)
test_x = test_x:cuda()
train_x = train_x:cuda()
myFile:close()
collectgarbage()

print(train_x:mean(1):sum() .. "--train data mean sum")
print(val_x:mean(1):sum() .. "--validation data mean sum")
print(test_x:mean(1):sum() .. "--test data mean sum")
print('train set size: ' .. train_x:size(1))
print('validation set size: ' .. val_x:size(1))
print('test set size: ' .. test_x:size(1))
print(train_y:size())

-- define deep neural networks
local hidden_nodes = opt.hidden_nodes
local model = nn.Sequential()
model:add(nn.Linear(train_x:size(2), hidden_nodes))
model:add(nn.ReLU())
hidden_layer_num = opt.hidden_layer_num
for i = 1,hidden_layer_num do
    model:add(nn.Dropout(opt.dropout))
    model:add(nn.Linear(hidden_nodes, hidden_nodes))
    model:add(nn.ReLU())
end
--model:add(nn.Dropout(opt.dropout))
model:add(nn.Linear(hidden_nodes, #classes))
model:add(nn.LogSoftMax())
-- initialize weights
--local method = 'xavier'
--local model = weight_init(model, method)
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
   momentum = 0.9
}

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

function train(epoch)
    -- set model to training mode
    -- model:training()
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
    print('train_accuracy: ' .. train_accuracy)
    trainLogger:add{['% mean accuracy (train set)'] = train_accuracy}
    if opt.plot then
        trainLogger:style{['% mean accuracy (train set)'] = '-'}
        trainLogger:plot()
    end
    
    -- save model
    --torch.save(opt.model_save .. '/model-' .. epoch .. '.t7', model)
    return train_err, train_accuracy
end

function val()
    collectgarbage()
    local current_loss = 0
    for i=1, val_x:size(1) do
        pred = model:forward(val_x[i])
        loss = criterion:forward(pred, val_y[i])
        current_loss = current_loss + loss
        confusion:add(pred, val_y[i])
    end
    val_err = current_loss/val_x:size(1)
    print(confusion)
    val_accuracy = confusion.totalValid * 100
    confusion:zero()
    --print('val_err: ' .. val_err)
    print('val_accuracy: ' .. val_accuracy)
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


for epoch = 1,100 do
    --[[if epoch % 10 == 0 then
        learning_params['learningRate'] = learning_params['learningRate'] / 1.1
        print('Learning rate changes to '.. learning_params['learningRate'])
        end--]]
    model:training()
    train_e[epoch] = train(epoch)

    model:evaluate()
    val_e[epoch], val_acc[epoch] = val()
    -- initialize the best model on val
    if epoch == 1 then
        best_val = val_acc[1]
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

local best_val_err, best_val_model = torch.min(torch.Tensor(val_e),1)
print('minimum validation err: ' .. best_val_err[1])
print('best model on validation set: ' .. best_val_model[1])
local best_train_err, best_train_model = torch.min(torch.Tensor(train_e), 1)
print('minimum train err: ' .. best_train_err[1])
print('best model on train set: ' .. best_train_model[1])

-- save the best model on val
torch.save(opt.model_save .. '/model-' .. best_val .. '.t7', best_model)


-- choose the best model for validation to test
--local model = torch.load(opt.model_save .. '/model-' .. best_val_model[1] .. '.t7')
model = best_model
model:evaluate()
test()
