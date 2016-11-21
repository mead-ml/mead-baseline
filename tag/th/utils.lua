require 'optim'
require 'xlua'

function newConv1D(ifm, ofm, filtsz, gpu)
   local tconv = gpu and cudnn.TemporalConvolution(ifm, ofm, filtsz) or nn.TemporalConvolution(ifm, ofm, filtsz)
   return tconv
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

function newSkipConn(seq, sz)
   local concat = nn.ConcatTable()
   concat:add(nn.Identity())
   local fconn = nn.Sequential()
   fconn:add(nn.Linear(sz, sz))
   fconn:add(activationFor("relu"))
   concat:add(fconn)
   local cadd = nn.CAddTable()
   seq:add(concat)
   seq:add(cadd)
end

function newLSTMCells(seq, input, output, layers, rnntype)

   from = input
   to = output
   for i=1,layers do

      if rnntype == 'blstm' then
	 local rnnfwd = nn.SeqLSTM(from, to)
	 rnnfwd.batchfirst = true
	 rnnfwd.maskzero = true
	 
	 local bwdseq = nn.Sequential()
	 bwdseq:add(nn.SeqReverseSequence(2))
	 local rnnbwd = nn.SeqLSTM(from, to)
	 rnnbwd.batchfirst = true
	 rnnbwd.maskzero = true
	 bwdseq:add(rnnbwd)
	 bwdseq:add(nn.SeqReverseSequence(2))

	 local concat = nn.ConcatTable()
	 concat:add(rnnfwd)
	 concat:add(bwdseq)
	 local cadd = nn.CAddTable()
	 seq:add(concat)
	 seq:add(cadd)
	 
      else
	 local lstm = nn.SeqLSTM(from, to)
	 lstm.batchfirst = true
	 lstm.maskzero = true
	 seq:add(lstm)
      end
      from = to

   end
   seq:add(nn.SplitTable(2, 3))
   return lstm
end

--[[
function newLSTMCells(seq, input, output, layers, rnntype)
   seq:add(nn.SplitTable(2))
   from = input
   to = output
   if rnntype == 'blstm' then
      local rnnfwd = nn.FastLSTM(from, to):maskZero(1)
      local rnnbwd = nn.FastLSTM(from, to):maskZero(1)
      seq:add(nn.BiSequencer(rnnfwd, rnnbwd, nn.CAddTable()))
    else
       print('Using FastLSTM (no matter what you asked for)')
       local rnnfwd = nn.FastLSTM(from, to):maskZero(1)
       seq:add(nn.Sequencer(rnnfwd))
   end
end
--]]

-- From the option list, pick one of [sgd, adagrad, adadelta, adam, rmsprop]
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
   elseif opt.optim == 'rmsprop' then
      optmeth = optim.rmsprop
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

-- Does this file exist
function exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

