--[[

  Data loader code for label followed by text TSVs.
  This actually supports
  <label>\s+<sentence>

--]]
function labelSent(line, lower)
   local labelText = line:split('%s+')
   if #labelText < 2 then return nil, nil end
   local label = labelText[1]
   local text = labelText[2]
   for i=3,#labelText do
      local w = labelText[i]
      if lower then
	 w = w:lower()
      end
      text = text .. ' ' .. w
   end
   return label,text
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

--[[
  Convert a table to a batch, where width is longest sentence,
  and shorter sentences are zero-padded at end.  
  Doesnt have to be full size if
  not enough data to fill: our code supports partial batches
--]]
function convertToBatch(tx, ty)
   local tlen = 0
   local dsz = tx[1]:size(2)
   for i=1,#tx do
      tlen = math.max(tlen, tx[i]:size(1))
   end

   local bx = torch.zeros(#tx, tlen, dsz)
   local by = torch.zeros(#ty)
   for i=1,#tx do
      bx[{i, {1,tx[i]:size(1)}}] = tx[i]
      by[i] = ty[i]
   end
   return bx, by
end

--[[
  Convert a table to a batch index, where width is longest sentence,
  and shorter sentences are padded using special token <PADDING>
  Doesnt have to be full size if
  not enough data to fill: our code supports partial batches
--]]

function convertToBatchIndices(w2v, tx, ty)
   local tlen = 0
   for i=1,#tx do
      local txi = tx[i]
      tlen = math.max(tlen, #txi)
   end

   local bx = {}
   local by = torch.zeros(#ty)
   for i=1,#tx do
      local txi = tx[i]
      
      local start = #txi + 1
      for j=start,tlen do
	 table.insert(txi, w2v.vocab['<PADDING>'])
      end

      table.insert(bx, txi)
      by[i] = ty[i]
   end
   return torch.Tensor(bx), by
end

--[[
  Build a vocab by processing all these files and generating a table
  of form {["hello"]=1, ..., ["world"]=1}
--]]
function buildVocab(files, lower)
    local vocab = {}

    for i=1,#files do
       local tsfile = io.open(files[i], 'r')

       for line in tsfile:lines() do  
	  
	  _, text = labelSent(line, lower)
	  
	  local toks = text:split(' ')
	  
	  local siglen = #toks
	  for i=1,siglen do
	     local w = toks[i]
	     vocab[w] = vocab[w] or 1
	  end
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
    local ts = {}
    local yt = {}
    local xt = {}
    local batchsz = options.batchsz or 1

    local vsz = w2v.vsz
    local dsz = w2v.dsz

    if options.w2cv then
       dsz = dsz + options.w2cv.dsz
    end
    options = options or {}

    local mxfiltsz = torch.max(torch.LongTensor(options.filtsz))
    local mxlen = options.mxlen or 1000
    -- for zeropadding the ends of the signal (AKA wide conv)
    local halffiltsz = math.floor(mxfiltsz / 2)

    -- Read in training data
    local tsfile = io.open(file, 'r')
    local linenum = 0

    local labelIdx = #f2i + 1


    local batchx = {}
    local batchy = {}
    local bx = nil
    local by = nil
    -- We will expand the data on demand for each batch (zero padding)
    for line in tsfile:lines() do  

       label, text = labelSent(line, options.lower)

       if label == nil then
	  print('Skipping invalid line ' .. line .. " " .. linenum)
	  
	  -- no continue's in lua :(
	  goto continue 
       end
       
       if f2i[label] == nil then
	  f2i[label] = labelIdx
	  labelIdx = labelIdx + 1
       end
       
       local y = torch.FloatTensor({f2i[label]})
       local toks = text:split(' ')
       
       local mx = math.min(#toks, mxlen)
       local siglen = mx + (2*halffiltsz)
       local x = torch.zeros(siglen, dsz)
       for i=1,mx do
	  local w = toks[i]
	  local z = w2v:lookup(w)
	  if options.w2cv then
	     local q = word2chvec(options.w2cv, w)
	     z = torch.cat(z, q)
	  end
	  x[{i + halffiltsz}] = z
       end
       linenum = linenum + 1
       
       table.insert(batchy, y)
       table.insert(batchx, x)
       
       if #batchy % batchsz == 0 then
	  -- evict the old batch, add a new one
	  bx, by = convertToBatch(batchx, batchy)
	  batchx = {}
	  batchy = {}
	  
	  table.insert(xt, bx)
	  table.insert(yt, by)
	  
       end
       
       ::continue::

    end
    
    if #batchx > 0 then
       -- evict the old batch, add a new one
       bx, by = convertToBatch(batchx, batchy)
       table.insert(xt, bx)
       table.insert(yt, by)
    end

    ts.y = yt
    ts.x = xt
    return ts, f2i
end

--[[
  Same code as above, but here we generate indices for sparse vectors
  Not fully formed dense representation feature vectors
  This is used by dynamic embedding (fine-tuned) CNN
--]]
function loadTemporalIndices(file, w2v, f2i, options)
    local ts = {}
    local yt = {}
    local xt = {}
    local batchsz = options.batchsz or 1

    local vsz = w2v.vsz
    local dsz = w2v.dsz

    options = options or {}
    local mxfiltsz = torch.max(torch.LongTensor(options.filtsz))
    local mxlen = options.mxlen or 1000
    -- for zeropadding the ends of the signal (AKA wide conv)
    local halffiltsz = math.floor(mxfiltsz / 2)

    -- Read in training data
    local tsfile = io.open(file, 'r')
    local linenum = 0

    local labelIdx = #f2i + 1


    local batchx = {}
    local batchy = {}
    local bx = nil
    local by = nil
    -- We will expand the data on demand for each batch (zero padding)
    for line in tsfile:lines() do  

       label, text = labelSent(line, options.lower)

       if label == nil then
	  print('Skipping invalid line ' .. line .. " " .. linenum)
	  
	  -- no continue's in lua :(
	  goto continue 
       end
       
       if f2i[label] == nil then
	  f2i[label] = labelIdx
	  labelIdx = labelIdx + 1
       end
       
       local y = torch.FloatTensor({f2i[label]})
       local toks = text:split(' ')
       
       local mx = math.min(#toks, mxlen)
       local siglen = mx + (2*halffiltsz)
       local xo = {}
       for i=1,halffiltsz do
	  table.insert(xo, w2v.vocab['<PADDING>'])
       end
       for i=1,mx do
	  local w = toks[i]
	  local key = w2v.vocab[w] or w2v.vocab['<PADDING>']
	  table.insert(xo, key)
       end
       for i=1,halffiltsz do
	  table.insert(xo, w2v.vocab['<PADDING>'])
       end
       linenum = linenum + 1
       
       table.insert(batchy, y)
       table.insert(batchx, xo)
       
       if #batchy % batchsz == 0 then
	  -- evict the old batch, add a new one
	  bx, by = convertToBatchIndices(w2v, batchx, batchy)
	  batchx = {}
	  batchy = {}
	  
	  table.insert(xt, bx)
	  table.insert(yt, by)
	  
       end
       
       ::continue::

    end
    
    if #batchx > 0 then
       -- evict the old batch, add a new one
       bx, by = convertToBatchIndices(w2v, batchx, batchy)
       table.insert(xt, bx)
       table.insert(yt, by)
    end

    ts.y = yt
    ts.x = xt
    return ts, f2i
end

