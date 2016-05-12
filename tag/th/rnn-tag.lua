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
DEF_PATIENCE = 10
DEF_RNN = 'blstm'
DEF_OPTIM = 'adadelta'
DEF_EPOCHS = 60
DEF_ETA = 0.4
DEF_HSZ = 100
DEF_PROC = 'gpu'
DEF_CLIP = 5
DEF_DECAY = 1e-7
DEF_MOM = 0.0
DEF_BATCHSZ = 8
torch.setdefaulttensortype('torch.FloatTensor')

--------------------------------------------------------------------
-- Create either an Elman-style RNN, an LSTM or a BLSTM using either
-- torch-rnn or ElementResearch rnn
-- If its the latter, we use a sequencer to have multiple copies of
-- Linear projection layer beneath the RNN.  If its the former, we
-- use a TemporalConvolution to weight share this layer
--------------------------------------------------------------------
function createTaggerModel(dsz, hsz, gpu, nc, rnntype)

    -- Create a processing chain
    local seq = nn.Sequential()


    -- Make RNN type
    --[[
       rnnmod = rnntype == 'blstm' and nn.SeqBRNN(dsz, hsz) or rnn.SeqLSTM(dsz, hsz)
       -- Force it to mask zeros
       rnnmod.maskzero = true
       seq:add(rnnmod)
       seq:add(nn.SplitTable(1))
    --]]
     if rnntype == 'blstm' then
	seq:add(nn.SplitTable(1))
	local rnnfwd = nn.FastLSTM(dsz, hsz):maskZero(1)
	local rnnbwd = nn.FastLSTM(dsz, hsz):maskZero(1)
	seq:add(nn.BiSequencer(rnnfwd, rnnbwd, nn.CAddTable()))
     else
	print('Using FastLSTM (no matter what you asked for)')
	local rnnfwd = nn.FastLSTM(dsz, hsz):maskZero(1)
	seq:add(nn.Sequencer(rnnfwd))
     end
     
     local subseq = nn.Sequential()
     subseq:add(nn.Dropout(0.5))
     subseq:add(newLinear(hsz, nc))
     subseq:add(nn.LogSoftMax())
     seq:add(nn.Sequencer(nn.MaskZero(subseq, 1)))

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

cmd:option('-batchsz', DEF_BATCHSZ, 'Batch size')
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
-- ElementResearch 'rnn' uses (T, B, H)
----------------------------------------
require 'nnx'
require 'rnn'

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
   opt.w2cv = w2cv
end

---------------------------------------
-- Load Feature Vectors
---------------------------------------
ts = conllSentsToVectors(opt.train, w2v, f2i, opt)
es = conllSentsToVectors(opt.eval, w2v, ts.f2i, opt)

local i2f = revlut(es.f2i)
local nc = #i2f
print('Number of classes ' .. nc)

---------------------------------------
-- Build model and criterion
---------------------------------------
local crit = createTaggerCrit(opt.gpu)
local model = createTaggerModel(dsz, opt.hsz, opt.gpu, nc, opt.rnn)

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
