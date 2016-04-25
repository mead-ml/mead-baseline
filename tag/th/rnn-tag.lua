--[[ 

This is an example of a using an RNN (plain RNN, LSTM or BSLTM) for tagging
using just word2vec pre-trained inputs (no fine-tuning).  It can optionally
use either ElementResearch's rnn package, or Justin Johnson's torch-rnn.

It's influenced by "Finding Function in Form: Compositional Character Models for Open Vocabulary Word Representation" - Ling et al, 2015, but here we take a quick and dirty approach to compositional character models, by just using a continuous bag of words average of letters.  The letter model is built using word2vec

--]]

require 'torch'
require 'nn'
require 'optim'
require 'tagutils'
require 'emb'
require 'ReversedCopy'
require 'train'
require 'data'
-----------------------------------------------------
-- Defaults if you dont specify args
-- Note that some arguments depend on which optim
-- is selected, and may be unused for others.  Try to
-- provide reasonable args for any algorithm a user selects
-----------------------------------------------------
DEF_TSF = './data/twpos-data-v0.3/oct27.splits/oct27.train'
DEF_ESF = './data/twpos-data-v0.3/oct27.splits/oct27.test'
DEF_EMBED = './data/GoogleNews-vectors-negative300.bin'
DEF_FILE_OUT = 'rnn-tagger.model'
DEF_PATIENCE = 6
DEF_RNN = 'blstm'
DEF_OPTIM = 'adadelta'
DEF_EPOCHS = 60
DEF_ETA = 0.32
DEF_HSZ = 100
DEF_PROC = 'gpu'
DEF_CLIP = 5
DEF_DECAY = 1e-7
DEF_MOM = 0.0

torch.setdefaulttensortype('torch.FloatTensor')

--------------------------------------------------------------------
-- Create either an Elman-style RNN, an LSTM or a BLSTM using either
-- torch-rnn or ElementResearch rnn
-- If its the latter, we use a sequencer to have multiple copies of
-- Linear projection layer beneath the RNN.  If its the former, we
-- use a TemporalConvolution to weight share this layer
--------------------------------------------------------------------
function createTaggerModel(dsz, hsz, gpu, nc, trnn, usernnpkg)

    -- Create a processing chain
    local seq = nn.Sequential()

    -- hidden unit depth
    local tsz = hsz

    if trnn == 'blstm' then
       -- For BSLTM twice as many hidden units
       tsz = 2 * hsz
    end

    if usernnpkg then
       -- RNN package makes sequencing very easy
       if trnn == 'blstm' then
	  seq:add(nn.SplitTable(1))
	  local rnnfwd = nn.FastLSTM(dsz, hsz)
	  local rnnbwd = nn.FastLSTM(dsz, hsz)
	  seq:add(nn.BiSequencer(rnnfwd, rnnbwd))
       else
	  print('Using Seq LSTM (no matter what you asked for)')
	  seq:add(nn.SeqLSTM(dsz, hsz))
	  seq:add(nn.SplitTable(1))
       end
       local subseq = nn.Sequential()
       subseq:add(nn.Dropout(0.5))
       subseq:add(nn.Linear(tsz, nc))
       seq:add(nn.Sequencer(subseq))
    else
       -- Use jcjohnson's torch-rnn
       if trnn == 'blstm' then
	  
	  -- Make two RNN units, one for forward direction, one for backward
	  local rnnfwd = nn.LSTM(dsz, hsz)
	  local rnnbwd = nn.LSTM(dsz, hsz)
	  
	  -- This will feed the same input, and will join
	  -- results along the 3rd dimension
	  local concat = nn.Concat(3)
	  
	  -- Add forward
	  concat:add(rnnfwd)
	  
	  -- Create a sub-chain for reverse
	  local subseq = nn.Sequential()
	  
	  -- Flip the signal so time is descending
	  subseq:add(nn.ReversedCopy())
	  subseq:add(rnnbwd)
	  
	  -- Flip the signal again when done
	  subseq:add(nn.ReversedCopy())
	  concat:add(subseq)
	  
	  -- Now add the BLSTM to the chain
	  seq:add(concat)
	  
       else
	  local rnn = trnn == 'lstm' and nn.LSTM(dsz, hsz) or nn.VanillaRNN(dsz, hsz)
	  seq:add(rnn)
       end
       
       -- Dropout before convolution
       seq:add(nn.Dropout(0.5))
       
       -- The convolution is now twice the depth due to the Concat
       -- The signal length is the same as before, and we are producing
       -- a depth of nc which allows us to predict output using shared weights
       seq:add(nn.TemporalConvolution(tsz, nc, 1))
    end

    -- GPU if possible
    return gpu and seq:cuda() or seq
end



--------------------------
-- Command line handling
--------------------------
local cmd = torch.CmdLine()
cmd:text('Parameters for RNN-based tagger')
cmd:text()
cmd:text('Options:')

cmd:option('-save', DEF_FILE_OUT, 'Save model to')
cmd:option('-rnn',  DEF_RNN)
cmd:option('-train', DEF_TSF, 'Training file')
cmd:option('-eval', DEF_ESF, 'Testing file')
cmd:option('-embed', DEF_EMBED, 'Word embeddings')
cmd:option('-cembed', 'none', 'Char embeddings')
cmd:option('-optim', DEF_OPTIM, 'Optimization methodx (sgd|adagrad|adam)')
cmd:option('-epochs', DEF_EPOCHS)
cmd:option('-eta', DEF_ETA)
cmd:option('-clip', DEF_CLIP)
cmd:option('-decay', DEF_DECAY)
cmd:option('-mom', DEF_MOM, 'Momentum for SGD')
cmd:option('-hsz', DEF_HSZ, 'Hidden layer units')
cmd:option('-proc', DEF_PROC)
cmd:option('-patience', DEF_PATIENCE)
cmd:option('-usernnpkg', false)

local opt = cmd:parse(arg)


----------------------------------------
-- Optimization
----------------------------------------

state, optmeth = optimMethod(opt)

--------------------------------------
-- Processing on GPU or CPU
----------------------------------------
opt.gpu = false
if opt.proc == 'gpu' then
   opt.gpu = true
   require 'cutorch'
   require 'cunn'
else
   opt.proc = 'cpu'
end
print('Processing on ' .. opt.proc)


----------------------------------------
-- RNN package to use
-- ElementResearch 'rnn' uses more common
-- (T, B, H)
-- torch-rnn uses (B, T, H)
----------------------------------------
if opt.usernnpkg then
   print('Using ElementResearch RNN package')
   require 'nnx'
   require 'rnn'
   opt.batch2ndDim = true
else
   require 'torch-rnn'
end


---------------------------------------
-- Load Word2Vec Model(s)
---------------------------------------
local f2i = {}
local w2v = Word2VecModel(opt.embed)
print('Loaded word embeddings: ' .. opt.embed)

local w2cv = nil

if opt.cembed ~= 'none' then
   w2cv = Word2VecModel(opt.cembed)
   print('Loaded char embeddings: ' .. opt.cembed)

end

-- If they included word and char vectors, use both
local dsz = w2v.dsz
if w2cv then
   dsz = dsz + w2cv.dsz
end

---------------------------------------
-- Load Feature Vectors
---------------------------------------
ts = conllSentsToVectors(opt.train, w2v, 0, f2i, w2cv)
es = conllSentsToVectors(opt.eval, w2v, 0, ts.f2i, w2cv)

local i2f = revlut(es.f2i)
local nc = #i2f
print('Number of classes ' .. nc)

---------------------------------------
-- Build model and criterion
---------------------------------------
local crit = createTaggerCrit(opt.gpu, opt.usernnpkg)
local model = createTaggerModel(dsz, opt.hsz, opt.gpu, nc, opt.rnn, opt.usernnpkg)

local errmin = 1;
local lastImproved = 0

for i=1,opt.epochs do
    print('Training epoch ' .. i)
    trainTaggerEpoch(crit, model, ts, optmeth, opt)
    confusion = optim.ConfusionMatrix(i2f)
    local erate = testTagger(model, es, crit, confusion, opt)
    if erate < errmin then
       lastImproved = i
       errmin = erate
       print('Lowest error achieved yet -- writing model')
       saveModel(model, opt.save, opt.gpu)
    end
    if (i - lastImproved) > opt.patience then
       print('Stopping due to persistent failures to improve')
       break
    end
end

print('Highest test acc: ' .. (100 * (1. - errmin)))
