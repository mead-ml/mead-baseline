require 'nn'
require 'optim'
require 'xlua'

function newConv1D(ifm, ofm, filtsz, gpu)
   local tconv = gpu and cudnn.TemporalConvolution(ifm, ofm, filtsz) or nn.TemporalConvolution(ifm, ofm, filtsz)
   return tconv
end

function writeTable(t, name)
   local df = torch.DiskFile(name, 'w')
   for k, v in pairs(t) do
      df:writeString(k .. '\t' .. v .. '\n')
   end
   df:close()
end

function activationFor(name, gpu)
   if name == 'ident' then
      return nn.Identity()
   elseif name == 'relu' then
      return gpu and cudnn.ReLU() or nn.ReLU()
   elseif name == 'lsoftmax' then
      return gpu and cudnn.LogSoftMax() or nn.LogSoftMax()
   end
   return gpu and cudnn.Tanh() or nn.Tanh()
end

function newLinear(inputSz, outputSz)
   local linear = nn.Linear(inputSz, outputSz)
   linear.weight:normal():mul(0.01)
   linear.bias:zero()
   return linear
end

-- From the option list, pick one of [sgd, adagrad, adadelta, adam]
function optimMethod(opt)

   print('Trying to use optim method: ' .. opt.optim)
   config = {
      learningRate = opt.eta,
      weightDecay = opt.decay
   }
   
   if opt.optim == 'sgd' then
      config.momentum = opt.mom
      optmeth = optim.sgd
   elseif opt.optim == 'adagrad' then
      optmeth = optim.adagrad
   elseif opt.optim == 'adadelta' then
      config.rho = 0.95
      config.eps = 1e-6
      optmeth = optim.adadelta
   elseif opt.optim == 'adam' then
      config.beta1 = opt.beta1 or 0.9
      config.beta2 = opt.beta2 or 0.999
      config.epsilon = opt.epsilon or 1e-8
      optmeth = optim.adam
   else
      print('Unknown optimization method ' .. opt.optim '. Using SGD with momentum')
      config.momentum = opt.mom
      opt.optim = 'sgd'
      optmeth = optim.sgd
   end
   return config, optmeth
end

-- Take a map[key] = index table, and make a map[index] = key
function revlut(f2i)
   local i2f = {}
   for k,v in pairs(f2i) do
      i2f[v] = k
   end
   return i2f
end

-- WIP
function saveModel(model, file, gpu)
   if gpu then model:float() end
   torch.save(file, model)
   if gpu then model:cuda() end
end

function loadModel(file, gpu)
   local model = torch.load(file)
   return gpu and model:cuda() or model
end

-- NaNs?
function hasNaN(t)
   mx = torch.max(t)
   return mx ~= mx
end

-- Does this file exist
function exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end
