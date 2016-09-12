require 'nn'
require 'optim'
require 'xlua'
require 'torchure'


function createSeq2SeqCrit(gpu)
   local crit = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1))
   return gpu and crit:cuda() or crit
end


function newLinear(inputSz, outputSz)
   local linear = nn.Linear(inputSz, outputSz)
   linear.weight:normal():mul(0.01)
   linear.bias:zero()
   return linear
end

function newLSTMCells(seq, input, output, layers)

   from = input
   to = output
   for i=1,layers do
      local lstm = nn.SeqLSTM(from, to)
      from = to
      lstm.maskzero = true
      seq:add(lstm)
   end
   seq:add(nn.SplitTable(1, 3))
   return lstm
end

-- https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua

function forwardConnect(model, len, layers)
   local encodes = model:get(1)
   local decodes = model:get(2)

   -- First module after LUT (FIXME for stacked LSTMs)
   start = 1
   for i=1,layers do
      j = start + i
      local encodesLSTM = encodes:get(j)
      local decodesLSTM = decodes:get(j)
      decodesLSTM.userPrevOutput = encodesLSTM.output[len]
      decodesLSTM.userPrevCell = encodesLSTM.cell[len]
   end
end

function backwardConnect(model, layers)

   local encodes = model:get(1)
   local decodes = model:get(2)

   -- First module after LUT (FIXME for stacked LSTMs)
   start = 1

   for i = 1,layers do
      j = start + i
      local decodesLSTM = decodes:get(j)
      local encodesLSTM = encodes:get(j)      
      encodesLSTM.userNextGradCell = decodesLSTM.userGradPrevCell
      encodesLSTM.gradPrevOutput = decodesLSTM.userGradPrevOutput
   end
end


function forgetModelState(model)
   local encodes = model:get(1)
   local decodes = model:get(2)
   encodes:forget()
   decodes:forget()
end

function getModelParameters(model)
   return model:getParameters()
end

--------------------------------------------------------------------
-- Create a seq2seq model, with connected decoder and encoder
--------------------------------------------------------------------
function createSeq2SeqModel(embed1, embed2, hsz, gpu, layers)

    -- Just holds both, makes it easier to read, write, update
    local model = nn.Container()

    local dsz = embed1.dsz
    -- Create a processing chain
    local decodes = nn.Sequential()
    local encodes = nn.Sequential()

    -- Lookup tables
    encodes:add(embed1)

    decodes:add(embed2)
    
    -- TODO: add stacking

    -- Cell added, and returned as well
    newLSTMCells(encodes, dsz, hsz, layers)
    encodes:add(nn.SelectTable(-1))

    newLSTMCells(decodes, dsz, hsz, layers)
    -- Encoder picks last output, passes this on


    -- Decoder also needs to pick and output
    local subseq = nn.Sequential()
    subseq:add(nn.Dropout(0.5))
    subseq:add(newLinear(hsz, embed2.vsz))
    subseq:add(nn.LogSoftMax())
    decodes:add(nn.Sequencer(nn.MaskZero(subseq, 1)))

     -- GPU if possible
    encodes = gpu and encodes:cuda() or encodes
    decodes = gpu and decodes:cuda() or decodes
    model:add(encodes)
    model:add(decodes)
    return model
end

-- From the option list, pick one of [sgd, adagrad, adadelta, adam]
function optimMethod(opt)

   print('Trying to use optim method: ' .. opt.optim)
   local optmeth = nil

   local state = {
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

