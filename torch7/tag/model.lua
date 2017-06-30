require 'torch'
require 'nn'
require 'optim'
require 'utils'
require 'torchure'
require 'ReformTBD'


-- Take in (T,B,Ch), spit out (T,B,D)
-- MapTable will operate the sequence over each word, which causes them
-- to be returned in (T,B,D) order
-- when we get to the char vectors, we have (B, Tch, Dch) outputs
-- This is what we need to do convolution over the words letters
function createConvCharWordEmbeddings(char_vec, opt)

   -- Each of these operations is performed starting at (B, Ch) over T clones
   local seq = nn.Sequential()
   
   local filts = opt.cfiltsz
   -- First step transforms one-hot to (B, Tch) to (B, Tch, Dch)
   seq:add(char_vec)
   concat = nn.Concat(2)
   for i=1,#filts do
      local filtsz = filts[i]
      print('Creating char filter of size ' .. filtsz)
      local subseq = nn.Sequential()
      subseq:add(newConv1D(char_vec.dsz, opt.wsz, filtsz, opt.gpu))
      subseq:add(activationFor("relu", opt.gpu))
      subseq:add(nn.Max(2))
      subseq:add(nn.Dropout(opt.pdrop))
      concat:add(subseq)
   end
   -- Concat leaves us with (B, D)
   seq:add(concat)

   newSkipConn(seq, #filts * opt.wsz)

   -- Sequencing leaves us with (T, B, D)
   local sequencer = nn.MapTable(seq)
   return sequencer

end

-- Take in (T, B, Ch) data, since we put it through a MapTable
-- we end up with each word being processed at a time, for each word
-- we are presented with (B, Ch) lookup table, which yields a char vector
-- at each T.  Then we sum along the Ch dimension which gives us a char word
-- vector
function createCBOWCharWordEmbeddings(char_vec, opt)

   local seq = nn.Sequential()
   
   seq:add(char_vec)

   -- To go through conv1d, we know that we have (B,T,H) data
   -- we want to sum over time
   seq:add(nn.Sum(2))
   local sequencer = nn.MapTable(seq)
   return sequencer

end

function createCharWordEmbeddings(char_vec, opt)
   if opt.cbow then
      print('Using continuous bag of characters for char-level word embeddings')
      return createCBOWCharWordEmbeddings(char_vec, opt)
   end
   print('Using CNN char-level word embeddings')
   return createConvCharWordEmbeddings(char_vec, opt)
end

function createTaggerModel(word_vec, char_vec, maxs, opt, nc)
    -- Create a processing chain
    local seq = nn.Sequential()
    local cfilts = opt.cfiltsz
    local gpu = opt.gpu
    local join_vec = word_vec.dsz + #cfilts * opt.wsz
    local par = nn.ParallelTable(1, 3)
    local parseq = nn.Sequential()
  
    -- This makes it (T, B, D) on input, and it rejoins along D in output
    parseq:add(nn.SplitTable(2,3))
    parseq:add(createCharWordEmbeddings(char_vec, opt))
    parseq:add(nn.JoinTable(1))
    -- (TxB, Dch)
    parseq:add(ReformTBD(maxs, #cfilts * opt.wsz))
    -- This puts us back into (B, T, Dch)
    parseq:add(nn.Transpose({1,2}))
    par:add(parseq)

    if opt.embed ~= 'NONE' then
       par:add(word_vec)
    else
       print('Direct word embeddings will not be used')
    end

    seq:add(par)
    -- This is b/c parallel joins the table on the third dimension
    seq:add(nn.JoinTable(3))
    -- This puts us back into (B, T, Dall)
    
    newLSTMCells(seq, join_vec, opt.hsz, opt.numrnn, opt.rnn)
    
    local subseq = nn.Sequential()
--    subseq:add(nn.Dropout(opt.pdrop))
--    newSkipConn(subseq, opt.hsz)
    subseq:add(nn.Dropout(opt.pdrop))

    subseq:add(newLinear(opt.hsz, nc))

    subseq:add(nn.LogSoftMax())
    seq:add(nn.MapTable(nn.MaskZero(subseq, 1)))

    -- GPU if possible
    return gpu and seq:cuda() or seq
end
