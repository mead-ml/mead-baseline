require 'utils'
require 'torchure'


VALID_TOKENS = revlut({
      'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z', 'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','(',')',',', '!','?',"'","`", '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'})


-- Bottom-up token cleanup
function doClean(l)
   
   local clean = ''
   -- preproc the line
   l,_ = l.gsub(l, "%'s", " 's ")
   l,_ = l.gsub(l, "%'ve", " 've ")
   l,_ = l.gsub(l, "n%'t", " n't ")
   l,_ = l.gsub(l, "%'re", " 're ")
   l,_ = l.gsub(l, "%'d", " 'd ")
   l,_ = l.gsub(l, "%'ll", " 'll ")
--   l,_ = l.gsub(l, ',', ' , ')
--   l,_ = l.gsub(l, '%?', ' ? ')
--   l,_ = l.gsub(l, '%!', ' ! ')
--   l,_ = l.gsub(l, '%(', ' ( ')
--   l,_ = l.gsub(l, '%)', ' ) ')
   
   for i=1,#l do
      local ch = l:sub(i, i)
      local append = VALID_TOKENS[ch] == nil and ' ' or ch
      clean = clean .. append
   end
-- not necessary, this gets resplit on 's+'
--   clean,_ = string.gsub(clean, '%s+', ' ')
   return clean
end

-- Average letter vectors over word
function word2chvec(w2cv, w)

    local csz = w2cv.dsz
    local wcv = torch.zeros(csz)
    local sf = 1./(#w + 1)
    for i=1,#w do
        local ch = w:sub(i, i)
        local z = w2cv:lookup(ch) * sf
	wcv = wcv + z
    end
    return wcv
end

function labelSent(line, clean, chars)

   local labelText = line:split('%s+')
   if #labelText < 2 then return nil, nil end
   local label = labelText[1]
   local text = ''
   for i=2,#labelText do
      local w = labelText[i]
      if clean then
        w = doClean(w:lower())
      end
      if chars then
	 wspc = ''
	 for j=1,#w do
	    local ch = w:sub(j, j)
	    text = text .. ' ' .. ch
	 end
      else
	 text = text .. ' ' .. w
      end
   end
   return label,text
end

--[[
  Build a vocab by processing all these files and generating a table
  of form {["hello"]=1, ..., ["world"]=1}
--]]
function buildVocab(files, clean, chars)
    local vocab = {}

    for i=1,#files do
       if files[i] ~= nil and files[i] ~= 'none' then
	  local tsfile = io.open(files[i], 'r')
	  
	  for line in tsfile:lines() do  
	     
	     _, text = labelSent(line, clean, chars)
	     
	     local toks = text:split(' ')
	     
	     local siglen = #toks
	     for i=1,siglen do
		local w = toks[i]
		vocab[w] = vocab[w] or 1
	     end
	  end
       else
	  print('Warn: skipping file ' .. files[i])
       end
    end
    return vocab
end

--[[ 

Load training/test data
This function does lazy mini-batching (using options.batchsz), and will fill partial patches.
For each minibatch at write time pick the max and generate a batch.  It also zero-pads the signal
appropriately for cross-correlation.  If there is a word-vector character model, it will average this
and concatenate the usual vector with the average character vector.  This is a very coarse approach,
but nevertheless helpful for cases where non-attested words are very frequent (e.g., Twitter),
and for morphological support (for things like tagging).

--]]

function loadTemporalEmb(file, w2v, f2i, options)
    local ts = options.ooc and FileBackedStore() or TableBackedStore()

    local batchsz = options.batchsz or 1

    local vsz = w2v.vsz
    local dsz = w2v.dsz

    if options.w2cv then
       dsz = dsz + options.w2cv.dsz
    end

    local mxfiltsz = torch.max(torch.LongTensor(options.filtsz))
    local mxlen = options.mxlen or 1000
    -- for zeropadding the ends of the signal (AKA wide conv)
    local halffiltsz = math.floor(mxfiltsz / 2)
    local labelIdx = #f2i + 1

    local n = numLines(file)
    local x = nil
    local y = nil
    local tsfile = io.open(file, 'r')
    local b = 0
    local i = 1
    -- We will expand the data on demand for each batch (zero padding)
    for line in tsfile:lines() do  

       label, text = labelSent(line, options.clean, options.chars)
       
       if f2i[label] == nil then
	  f2i[label] = labelIdx
	  labelIdx = labelIdx + 1
       end

       local offset = (i - 1) % batchsz
       
       if offset == 0 then
	  if b > 0 then
	     ts:put({x=x,y=y})
	  end
	  b = b + 1
	  thisBatchSz = math.min(batchsz, n - i + 1)
	  x = torch.FloatTensor(thisBatchSz, mxlen + mxfiltsz, dsz):fill(0)
	  y = torch.LongTensor(thisBatchSz):fill(0)
       end
       y[offset+1] = f2i[label]
       local toks = text:split(' ')
       
       local mx = math.min(#toks, mxlen)
       for j=1,mx do
	  local w = toks[j]
	  local z = w2v:lookup(w)
	  if options.w2cv then
	     local q = word2chvec(options.w2cv, w)
	     z = torch.cat(z, q)
	  end
	  x[{offset+1, j+halffiltsz}] = z
       end

       i = i + 1
    end

    if thisBatchSz > 0 then
       -- evict the old batch, add a new one
       ts:put({x=x,y=y})
    end

    return ts, f2i
end


-- Create a valid split of this data store, splitting on a fraction
function validSplit(dataStore, splitfrac, ooc)
   local train = ooc and FileBackedStore() or TableBackedStore()
   local valid = ooc and FileBackedStore() or TableBackedStore()
   local numinst = dataStore:size()
   local heldout = numinst * splitfrac
   local holdidx = numinst - heldout
   
   for i=1,numinst do
      local txy = dataStore:get(i)
      if i < holdidx then
	 train:put(txy)
      else
	 valid:put(txy)
      end
      
   end
   
   return train, valid
end

function loadTemporalIndices(file, w2v, f2i, options)
    local ts = options.ooc and FileBackedStore() or TableBackedStore()
    local batchsz = options.batchsz or 1

    local vsz = w2v.vsz
    local dsz = w2v.dsz

    local PAD = w2v.vocab['<PADDING>']
    local mxfiltsz = torch.max(torch.LongTensor(options.filtsz))
    local mxlen = options.mxlen or 1000
    -- for zeropadding the ends of the signal (AKA wide conv)
    local halffiltsz = math.floor(mxfiltsz / 2)
    local labelIdx = #f2i + 1

    local n = numLines(file)
    local x = nil
    local y = nil
    local tsfile = io.open(file, 'r')
    local b = 0
    local i = 1

    -- We will expand the data on demand for each batch (zero padding)
    for line in tsfile:lines() do  

       label, text = labelSent(line, options.clean, options.chars)
       
       if f2i[label] == nil then
	  f2i[label] = labelIdx
	  labelIdx = labelIdx + 1
       end


       local offset = (i - 1) % batchsz
       
       if offset == 0 then
	  if b > 0 then
	     ts:put({x=x,y=y})
	  end
	  b = b + 1
	  thisBatchSz = math.min(batchsz, n - i + 1)
	  x = torch.LongTensor(thisBatchSz, mxlen + mxfiltsz):fill(PAD)
	  y = torch.LongTensor(thisBatchSz):fill(0)
       end

       y[offset+1] = f2i[label]
       local toks = text:split(' ')
       
       local mx = math.min(#toks, mxlen)
       for j=1,mx do
	  local w = toks[j]
	  local key = w2v.vocab[w] -- or w2v.vocab['<PADDING>']
	  x[{offset+1,j+halffiltsz}] = key
       end
       i = i + 1
    end
    
    if thisBatchSz > 0 then
       -- evict the old batch, add a new one
       ts:put({x=x,y=y})
    end

    return ts, f2i
end

