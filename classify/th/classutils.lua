require 'nn'
require 'optim'
require 'xlua'

function activationFor(name)
   if name == 'ident' then
      return nn.Identity()
   elseif name == 'relu' then
      return nn.ReLU()
   end
   return nn.Tanh()
end

-- From the option list, pick one of [sgd, adagrad, adadelta, adam]
function optimMethod(opt)

   print('Trying to use optim method: ' .. opt.optim)
   state = {
      learningRate = opt.eta,
      weightDecay = opt.decay
   }
   
   if opt.optim == 'sgd' then
      state.momentum = opt.mom
      optmeth = optim.sgd
   elseif opt.optim == 'adagrad' then
      optmeth = optim.adagrad
   elseif opt.optim == 'adadelta' then
      state.rho = 0.95
      state.eps = 1e-6
      optmeth = optim.adadelta
   elseif opt.optim == 'adam' then
      state.beta1 = opt.beta1 or 0.9
      state.beta2 = opt.beta2 or 0.999
      state.epsilon = opt.epsilon or 1e-8
      optmeth = optim.adam
   else
      print('Unknown optimization method ' .. opt.optim '. Using SGD with momentum')
      state.momentum = opt.mom
      opt.optim = 'sgd'
      optmeth = optim.sgd
   end
   return state, optmeth
end

-- Take a map[key] = index table, and make a map[index] = key
function revlut(f2i)
   local i2f = {}
   for k,v in pairs(f2i) do
      i2f[tonumber(v)] = k
   end
   return i2f
end

-- WIP
function saveModel(model, file, gpu)
   if gpu then model:float() end
   torch.save(file, model)
   if gpu then model:cuda() end
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
