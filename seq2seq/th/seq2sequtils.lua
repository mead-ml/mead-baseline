require 'nn'
require 'optim'
require 'xlua'

-- Take a map[key] = index table, and make a map[index] = key
function revlut(f2i)
   local i2f = {}
   for k,v in pairs(f2i) do
      i2f[v] = k
   end
   return i2f
end

function revtab(tab)
    local size = #tab
    local newTable = {}

    for i,v in ipairs ( tab ) do
        newTable[size-i] = v
    end
    return newTable
end

function lookupSent(rlut, lu, rev)
   
   local words = {}
   for i=1,lu:size(1) do
      local word = rlut[lu[i]]
      if word == '<EOS>' then
	 table.insert(words, word)
	 break
      end
      if word ~= '<PADDING>' then
	 table.insert(words, word)
      end
   end
   if rev then
      words = revtab(words)
   end
   return table.concat(words, " ")
end

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


function newBLSTMCell(seq, input, output)

   local blstm = nn.SeqBRNN(input, output, false, nn.CAddTable())
   seq:add(nn.SplitTable(1))
   seq:add(blstm)
   return blstm
end

function newLSTMCell(seq, input, output)
   local lstm = nn.SeqLSTM(input, output)
   lstm.maskzero = true
   seq:add(lstm)
   seq:add(nn.SplitTable(1, 3))
   return lstm
end

-- https://github.com/Element-Research/rnn/blob/master/examples/encoder-decoder-coupling.lua

function forwardConnect(model, len)
   local encodes = model:get(1)
   local decodes = model:get(2)

   -- First module after LUT (FIXME for stacked LSTMs)
   local encodesLSTM = encodes:get(2)
   local decodesLSTM = decodes:get(2)
   decodesLSTM.userPrevOutput = nn.rnn.recursiveCopy(decodesLSTM.userPrevOutput, encodesLSTM.output[len])
   decodesLSTM.userPrevCell = nn.rnn.recursiveCopy(decodesLSTM.userPrevCell, encodesLSTM.cell[len])
end

function backwardConnect(model, len)

   local encodes = model:get(1)
   local decodes = model:get(2)

   -- First module after LUT (FIXME for stacked LSTMs)
   local encodesLSTM = encodes:get(2)
   local decodesLSTM = decodes:get(2)

   encodesLSTM.userNextGradCell = nn.rnn.recursiveCopy(encodesLSTM.userNextGradCell, decodesLSTM.userGradPrevCell)
   encodesLSTM.gradPrevOutput = nn.rnn.recursiveCopy(encodesLSTM.gradPrevOutput, decodesLSTM.userGradPrevOutput)
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
function createSeq2SeqModel(embed1, embed2, hsz, gpu)

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
    newLSTMCell(encodes, dsz, hsz)
    encodes:add(nn.SelectTable(-1))

    newLSTMCell(decodes, dsz, hsz)
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

function saveModel(model, file, gpu)
   if gpu then model:float() end
   torch.save(file, model)
   if gpu then model:cuda() end
end
