--[[ 

Using seq2seq to learn a mapping between thought vectors.  This can be used
with parallel data or conversation pairs.  File format is a tab separated sentence to encode, and sentence to decode.

Here is a sample line where encoder is built for japanese, and decoder is
built for english

ムーリエル は ２０歳 に なりました 。	Muiriel is 20 now .

The data in the first sentence is read in and an <EOS> token is appended.
The decoded sentence is read in forward order.

A lookup table is used (pretrained embedddings) on input to project this model

The strategy used for this is to provide a separate 
LookupTable for each end of the data.  This makes it possible to load pretrained data in each domain, or to use the same embeddingson each end with fine-tuning separate


--]]

require 'torch'
require 'nn'
require 'optim'
require 'seq2sequtils'
require 'emb'
require 'train'
require 'data'
-----------------------------------------------------
-- Defaults if you dont specify args
-- Note that some arguments depend on which optim
-- is selected, and may be unused for others.  Try to
-- provide reasonable args for any algorithm a user selects
-----------------------------------------------------
DEF_TSF = '/data/pairs-train.txt'
DEF_ESF = '/data/pairs-test.txt'
DEF_EMBED1 = '/data/xdata/oct-s140clean-uber.cbow-bin'
DEF_EMBED2 = '/data/xdata/oct-s140clean-uber.cbow-bin'
DEF_FILE_OUT = 'seq2seq.model'
DEF_PATIENCE = 10
DEF_OPTIM = 'adadelta'
DEF_EPOCHS = 60
DEF_ETA = 0.4
DEF_HSZ = 100
DEF_PROC = 'gpu'
DEF_CLIP = 5
DEF_DECAY = 1e-7
DEF_MOM = 0.0
DEF_BATCHSZ = 8
DEF_EMBUNIF = 0.25
DEF_SAMPLE = false
torch.setdefaulttensortype('torch.FloatTensor')

--------------------------
-- Command line handling
--------------------------
local cmd = torch.CmdLine()
cmd:text('Parameters for seq2seq')
cmd:text()
cmd:text('Options:')

cmd:option('-batchsz', DEF_BATCHSZ, 'Batch size')
cmd:option('-save', DEF_FILE_OUT, 'Save model to')
cmd:option('-train', DEF_TSF, 'Training file')
cmd:option('-eval', DEF_ESF, 'Testing file')
cmd:option('-embed1', DEF_EMBED1, 'Word embeddings (1)')
cmd:option('-embed2', DEF_EMBED2, 'Word embeddings (2)')
cmd:option('-embunif', DEF_EMBUNIF, 'Word2Vec initialization for non-attested attributes')
cmd:option('-optim', DEF_OPTIM, 'Optimization methodx (sgd|adagrad|adam|adadelta)')
cmd:option('-epochs', DEF_EPOCHS)
cmd:option('-eta', DEF_ETA)
cmd:option('-clip', DEF_CLIP)
cmd:option('-decay', DEF_DECAY)
cmd:option('-mom', DEF_MOM, 'Momentum for SGD')
cmd:option('-hsz', DEF_HSZ, 'Hidden layer units')
cmd:option('-proc', DEF_PROC)
cmd:option('-patience', DEF_PATIENCE)
cmd:option('-sample', DEF_SAMPLE, 'Perform sampling to find candidate decodes')
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

local vocab1 = buildVocab(1, {opt.train, opt.eval})
local vocab2 = buildVocab(2, {opt.train, opt.eval})

local embed1 = Word2VecLookupTable(opt.embed1, vocab1, opt.embunif)
print('Loaded word embeddings: ' .. opt.embed1)

local embed2 = Word2VecLookupTable(opt.embed2, vocab2, opt.embunif)
print('Loaded word embeddings: ' .. opt.embed2)

function afterhook() 
      embed1.weight[1]:zero()
      embed2.weight[1]:zero()
end

opt.afteroptim = afterhook

---------------------------------------
-- Load Feature Vectors
---------------------------------------
ts = sentsToIndices(opt.train, embed1, embed2, opt)
es = sentsToIndices(opt.eval, embed1, embed2, opt)
local rlut1 = revlut(embed1.vocab)
local rlut2 = revlut(embed2.vocab)
---------------------------------------
-- Build model and criterion
---------------------------------------
local crit = createSeq2SeqCrit(opt.gpu)
local model = createSeq2SeqModel(embed1, embed2, opt.hsz, opt.gpu)

local errmin = 1000;
local lastImproved = 0

-- If you show this it'll be random
-- showBatch(model, es, rlut1, rlut2, embed2, opt)

for i=1,opt.epochs do
    print('Training epoch ' .. i)
    trainSeq2SeqEpoch(crit, model, ts, optmeth, opt)
    local erate = testSeq2Seq(model, es, crit, opt)
    if erate < errmin then
       lastImproved = i
       errmin = erate
       showBatch(model, es, rlut1, rlut2, embed2, opt)


       print('Lowest error achieved yet -- writing model')
       saveModel(model, opt.save, opt.gpu)
    end
    if (i - lastImproved) > opt.patience then
       print('Stopping due to persistent failures to improve')
       break
    end
end

print('Highest test acc: ' .. (100 * (1. - errmin)))
