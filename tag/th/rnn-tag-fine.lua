--[[ 

This is an example of a using an RNN (plain RNN, LSTM or BSLTM) for tagging
using just word2vec pre-trained inputs with fine-tuning.  It can optionally
use either ElementResearch's rnn package, or Justin Johnson's torch-rnn.

--]]

require 'torch'
require 'nn'
require 'optim'
require 'utils'
require 'torchure'
require 'train'
require 'data'

-----------------------------------------------------
-- Defaults if you dont specify args
-- Note that some arguments depend on which optim
-- is selected, and may be unused for others.  Try to
-- provide reasonable args for any algorithm a user selects
-----------------------------------------------------
DEF_BATCHSZ = 8
DEF_TSF = './data/twpos-data-v0.3/oct27.splits/oct27.train'
DEF_ESF = './data/twpos-data-v0.3/oct27.splits/oct27.test'
DEF_EMBED = './data/GoogleNews-vectors-negative300.bin'
DEF_FILE_OUT = 'rnn-tagger.model'
DEF_PATIENCE = 10
DEF_RNN = 'blstm'
DEF_OPTIM = 'adadelta'
DEF_EPOCHS = 60
DEF_ETA = 0.32
DEF_HSZ = 100
DEF_PROC = 'gpu'
DEF_CLIP = 5
DEF_DECAY = 1e-7
DEF_MOM = 0.0
DEF_EMBUNIF = 0.25
DEF_OUT_OF_CORE = false
torch.setdefaulttensortype('torch.FloatTensor')


--------------------------------------------------------------------
-- Create either an Elman-style RNN, an LSTM or a BLSTM using either
-- torch-rnn or ElementResearch rnn
-- If its the latter, we use a sequencer to have multiple copies of
-- Linear projection layer beneath the RNN.  If its the former, we
-- use a TemporalConvolution to weight share this layer
--------------------------------------------------------------------
function createTaggerModel(w2v, hsz, gpu, nc, rnntype, pdrop)

    -- Create a processing chain
    local seq = nn.Sequential()
    local dsz = w2v.dsz

    seq:add(w2v)
    seq:add(nn.SplitTable(1))

    if rnntype == 'blstm' then
       
       local rnnfwd = nn.FastLSTM(dsz, hsz):maskZero(1)
       local rnnbwd = nn.FastLSTM(dsz, hsz):maskZero(1)
       seq:add(nn.BiSequencer(rnnfwd, rnnbwd, nn.CAddTable()))
    else
       print('Using FastLSTM (no matter what you asked for)')
       local rnnfwd = nn.FastLSTM(dsz, hsz):maskZero(1)
       seq:add(nn.Sequencer(rnnfwd))
    end
    
    local subseq = nn.Sequential()
    subseq:add(nn.Dropout(pdrop))
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
cmd:option('-batchsz', DEF_BATCHSZ, 'Batch size')
cmd:option('-save', DEF_FILE_OUT, 'Save model to')
cmd:option('-rnn', DEF_RNN)
cmd:option('-train', DEF_TSF, 'Training file')
cmd:option('-eval', DEF_ESF, 'Testing file')
cmd:option('-embed', DEF_EMBED, 'Word embeddings')
cmd:option('-embunif', DEF_EMBUNIF, 'Word2Vec initialization for non-attested attributes')
cmd:option('-optim', DEF_OPTIM, 'Optimization methodx (sgd|adagrad|adadelta|adam)')
cmd:option('-epochs', DEF_EPOCHS)
cmd:option('-eta', DEF_ETA)
cmd:option('-clip', DEF_CLIP)
cmd:option('-decay', DEF_DECAY)
cmd:option('-mom', DEF_MOM, 'Momentum for SGD')
cmd:option('-hsz', DEF_HSZ, 'Hidden layer units')
cmd:option('-proc', DEF_PROC)
cmd:option('-patience', DEF_PATIENCE)
-- Strongly recommend its set to 'true' for non-massive GPUs
cmd:option('-keepunused', false, 'Keep unattested words in Lookup Table')
cmd:option('-pdrop', 0.5, 'Dropout probability')
cmd:option('-ooc', DEF_OUT_OF_CORE, 'Should data batches be file-backed?')

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

------------------------------------------------------------------------
-- This option is to clip unattested features from the LookupTable, for
-- processing efficiency
-- Reading from eval is not really cheating here
-- We are just culling the set to let the LUT be more compact for tests
-- This data already existed in pre-training!  We are just being optimal
-- here to keep memory footprint small
------------------------------------------------------------------------
local vocab = nil

if opt.keepunused == false then
   vocab = conllBuildVocab({opt.train, opt.eval})
   print('Removing unattested words')
end
---------------------------------------
-- Load Word2Vec Model(s)
---------------------------------------
local f2i = {}
local w2v = Word2VecLookupTable(opt.embed, vocab, opt.embunif)

print('Loaded word embeddings: ' .. opt.embed)

function afterhook() 
      w2v.weight[1]:zero()
end

opt.afteroptim = afterhook

---------------------------------------
-- Load Feature Vectors
---------------------------------------
ts,f2i = conllSentsToIndices(opt.train, w2v, f2i, opt)
es,f2i = conllSentsToIndices(opt.eval, w2v, f2i, opt)

local i2f = revlut(f2i)
local nc = #i2f
print('Number of classes ' .. nc)

---------------------------------------
-- Build model and criterion
---------------------------------------
local crit = createTaggerCrit(opt.gpu)
local dsz = w2v.dsz
local model = createTaggerModel(w2v, opt.hsz, opt.gpu, nc, opt.rnn, opt.pdrop)

local errmin = 1;
local lastImproved = 0

for i=1,opt.epochs do
    print('Training epoch ' .. i)
    confusion = optim.ConfusionMatrix(i2f)
    trainTaggerEpoch(crit, model, ts, optmeth, opt)
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
