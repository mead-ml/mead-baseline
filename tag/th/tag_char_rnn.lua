require 'torch'
require 'nn'
require 'optim'
require 'utils'
require 'torchure'
require 'train'
require 'data'
require 'cudnn'
require 'model'


DEF_BATCHSZ = 50
DEF_TSF = './data/twpos-data-v0.3/oct27.splits/oct27.train'
DEF_VSF = './data/twpos-data-v0.3/oct27.splits/oct27.dev'
DEF_ESF = './data/twpos-data-v0.3/oct27.splits/oct27.test'
DEF_FILE_OUT = 'rnn-tagger.model'
DEF_EVAL_OUT = 'rnn-tagger-test.txt'
DEF_PATIENCE = 10
DEF_RNN = 'blstm'
DEF_NUM_RNN = 1
DEF_OPTIM = 'adadelta'
DEF_EPOCHS = 60
DEF_ETA = 0.32
DEF_CFILTSZ = '{1,2,3,4,5,7}'
DEF_HSZ = 100
DEF_CHARSZ = 16
DEF_WSZ = 50
DEF_PROC = 'gpu'
DEF_CLIP = 5
DEF_DECAY = 1e-7
DEF_MOM = 0.0
DEF_UNIF = 0.25
DEF_PDROP = 0.5
DEF_MXLEN = 40
DEF_VALSPLIT = 0.15
DEF_OUT_OF_CORE = false
DEF_EMBED = 'NONE'
DEF_CEMBED = 'NONE'
torch.setdefaulttensortype('torch.FloatTensor')

--------------------------
-- Command line handling
--------------------------
local cmd = torch.CmdLine()
cmd:option('-cembed', DEF_CEMBED, 'Character-level pre-trained embeddings')
cmd:option('-embed', DEF_EMBED, 'Word-level pre-trained embeddings')
cmd:option('-batchsz', DEF_BATCHSZ, 'Batch size')
cmd:option('-save', DEF_FILE_OUT, 'Save model to')
cmd:option('-conll_output', DEF_EVAL_OUT, 'Place to put test CONLL file')
cmd:option('-rnn', DEF_RNN, 'RNN type (blstm|lstm) default is blstm')
cmd:option('-numrnn', DEF_NUM_RNN, 'The depth of stacked RNNs')
cmd:option('-train', DEF_TSF, 'Training file')
cmd:option('-valid', DEF_VSF, 'Validation file (optional)')
cmd:option('-eval', DEF_ESF, 'Testing file')
cmd:option('-valsplit', DEF_VALSPLIT, 'Fraction training used for validation if no set is given')
cmd:option('-unif', DEF_UNIF, 'Vocab embedding initialization for non-attested attributes')
cmd:option('-optim', DEF_OPTIM, 'Optimization methodx (sgd|adagrad|adadelta|adam)')
cmd:option('-epochs', DEF_EPOCHS)
cmd:option('-eta', DEF_ETA)
cmd:option('-clip', DEF_CLIP)
cmd:option('-decay', DEF_DECAY)
cmd:option('-mom', DEF_MOM, 'Momentum for SGD')
cmd:option('-cfiltsz', DEF_CFILTSZ, 'Convolution filter width')
cmd:option('-hsz', DEF_HSZ, 'Hidden layer units')
cmd:option('-charsz', DEF_CHARSZ, 'Character embedding depth')
cmd:option('-wsz', DEF_WSZ, 'Word embedding depth')
cmd:option('-proc', DEF_PROC)
cmd:option('-patience', DEF_PATIENCE)
cmd:option('-dropout', DEF_PDROP, 'Dropout probability')
cmd:option('-ooc', DEF_OUT_OF_CORE, 'Should data batches be file-backed?')
cmd:option('-mxlen', DEF_MXLEN, 'Max sentence length')
cmd:option('-cbow', false, 'Do CBOW for characters')
local opt = cmd:parse(arg)
if opt.cbow then
   opt.cfiltsz = "{1}"
end
opt.cfiltsz = loadstring("return " .. opt.cfiltsz)()

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

require 'nnx'
require 'rnn'

local vocab = nil

maxw, vocab_ch, vocab_word = conllBuildVocab({opt.train, opt.eval, opt.valid})

maxw = math.min(maxw, opt.mxlen)
print('Max word length ' .. maxw)
---------------------------------------
-- Load Word2Vec Model(s)
---------------------------------------
local f2i = {}
local finetune = true

if opt.embed ~= 'NONE' then
   print('Loading word embeddings ' .. opt.embed)
   word_vec = Word2VecLookupTable(opt.embed, vocab_word, opt.unif, false, finetune)
else
   word_vec = {}
   word_vec.dsz = 0
end

-- Character embeddings
if opt.cembed == 'NONE' then
   if opt.charsz ~= opt.wsz and opt.cbow == true then
      print('Warning, you have opted for CBOW char embeddings, but have provided differing sizes for char embedding depth and word depth.  This is not possible, forcing char embedding depth to be word depth ' .. opt.wsz)
      opt.charsz = opt.wsz
   end

   wch_vec = VocabLookupTable(vocab_ch, opt.charsz, opt.unif)
else
   print('Using pre-trained character embeddings ' .. opt.cembed)
   wch_vec = Word2VecLookupTable(opt.cembed, vocab_ch, opt.unif, false, finetune)
   opt.charsz = wch_vec.dsz
   if opt.charsz ~= opt.wsz and opt.cbow == true then
      print('Warning, you have opted for CBOW char embeddings, and have provided pre-trained char vector embeddings.  To make this work, setting word vector size to character vector size ' .. opt.charsz)
      opt.wsz = opt.charsz
   end

end

--print(word_vec.vocab["<PADDING>"])
function afterhook() 
   wch_vec.weight[1]:zero()
   if word_vec.weight then
      word_vec.weight[1]:zero()
   end
end
opt.afteroptim = afterhook

-- Load Feature Vectors
---------------------------------------
ts,f2i,_ = conllSentsToIndices(opt.train, word_vec, wch_vec, maxw, f2i, opt)
print('Loaded training data')

if opt.valid ~= 'NONE' then
   print('Using provided validation data')
   vs,f2i,_ = conllSentsToIndices(opt.valid, word_vec, wch_vec, maxw, f2i, opt)
else
   ts,vs = validSplit(ts, opt.valsplit, opt.ooc)
   print('Created validation split')
end

es,f2i,txts = conllSentsToIndices(opt.eval, word_vec, wch_vec, maxw, f2i, opt)
-- print(txts)
print(f2i)
print('Using ' .. ts:size() .. ' examples for training')
print('Using ' .. vs:size() .. ' examples for validation')
print('Using ' .. es:size() .. ' examples for test')

local i2f = revlut(f2i)
local nc = #i2f
print('Number of classes ' .. nc)
---------------------------------------
-- Build model and criterion
---------------------------------------
local crit = createTaggerCrit(opt.gpu)


local model = createTaggerModel(word_vec, wch_vec, opt, nc)

local maxacc = 0
local lastImproved = 0

for i=1,opt.epochs do
    print('Training epoch ' .. i)
    
    trainTaggerEpoch(crit, model, ts, optmeth, opt)
    local loss, acc = testTagger('Validation', model, vs, crit, i2f, opt)

    if acc > maxacc then
       lastImproved = i
       maxacc = acc
       print('Highest dev acc achieved yet -- writing model')
       saveModel(model, opt.save, opt.gpu)
    end
    if (i - lastImproved) > opt.patience then
       print('Stopping due to persistent failures to improve')
       break
    end
end

print("-----------------------------------------------------")
print('Highest dev acc: ' .. maxacc)
print('=====================================================')
print('Evaluating best model on test data')

model = loadModel(opt.save, opt.gpu)
print('Reloaded best model')

-- For final evaluation, dont batch, want to get all examples once
-- Also this will write out each example to the opt.output file, which can
-- be used with conlleval.pl to test overall performance
opt.batchsz = 1
local loss, acc = testTagger('Test', model, es, crit, i2f, opt, txts)

print("-----------------------------------------------------")
print('Test loss '.. loss)
print('Test acc ' .. acc)
print('=====================================================')
