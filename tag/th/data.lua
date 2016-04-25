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
  Try multiple ways to get the word.  Unlike in text
  categorization, here we really need something for most
  words to not have big gaps in data
--]]
function tryGetWord(w2v, word)
   z = w2v:lookup(word, true)
   if z == nil then
      return w2v:lookup(word:lower(), true)
   else
      return z
   end

end


--[[
  Total hack: FIXME!
  To make this work, I replaced some samples with special tokens prior to
  running word2vec.  This is fairly specific code below for producing
  Twitter taggers using word2vec embeddings.
--]] 
function newWord(word)
   if word:match('^http') then
      return 'URL'
   elseif word:match('^@') then
      return '@@@@'
   elseif word:match('^#') then
      return '####'

   elseif word == '"' then
      return ','
   elseif word == ':)' or word == ':(((' or word == ':D' or word == '=)' or word == ':-)' or word == '=(' then
      return ';)'
   elseif word == '<3' then
      return '&lt;3'
   elseif word:match('^[0-9]+$') then
      return '0000';
   elseif word:match('^[A-Z]') and word:match('\'s') then
      return 'John\'s'
   else
      return word
   end
end

-- Get the vector for the token trying many things
function vecFor(w2v, tok)
   local non = 0

   z = tryGetWord(w2v, tok)

   if z == nil then

      local alt = newWord(tok) -- w2v:lookup(tok, true)
      z = w2v:lookup(alt, true)

      if z == nil then
	 non = non + 1.0
	 z = torch.zeros(w2v.dsz):float()
      end
   end
   return z, non
end

--[[
  Build a vocab by processing all these files and generating a table
  of form {["hello"]=1, ..., ["world"]=1}
--]]
function conllBuildVocab(files)
    local vocab = {['URL']=1,
                   ['@@@@']=1,
		   ['####']=1,
		   [',']=1,
		   [';)']=1,
		   ['&lt;3']=1,
		   ['0000']=1,
		   ['John\'s']=1}

    for i=1,#files do
       local tsfile = io.open(files[i], 'r')
       for line in tsfile:lines() do

	  local states = line:split('%s+')
	  if #states ~= 0 then

	      local w = states[1]
	      -- Allow either case for backoff
	      vocab[w] = vocab[w] or 1
	      w = w:lower()
	      vocab[w] = vocab[w] or 1
	  end
       end
    end

    return vocab
end


--[[
  Read in CONLL data and produce an index for each word.  Used for
  fine-tuning taggers
--]]
function conllSentsToIndices(file, w2v, zp, f2i, options)
   
    local tsfile = io.open(file, 'r')
    local linenum = 1

    local lbl = {}
    local lbls = {}
    local ts = {}
    local txt = {}
    local txts = {}
    local xt = {}
    local yt = {}
    local x = {}
    local y = {}
    local idx = 0
    local wsz = w2v.dsz
    print('Word vector sz ' .. wsz)
    for line in tsfile:lines() do
       local state = line:split('%s+')
       if #state == 0 then
	  -- time to return the set
	  
	  table.insert(txts, txt)
	  table.insert(lbls, lbl)
	  table.insert(yt, torch.Tensor(y))
	  txt = {}
	  lbl = {}
	  y = {}
       else
	  local label = state[#state]
	  if not f2i[label] then
	     idx = idx + 1
	     f2i[label] = idx
	     -- print('Label ' .. label)
	  end

	  local word = state[1]
	  table.insert(txt, word)
	  table.insert(lbl, label)
	  table.insert(y, f2i[label])
       end
    end

    local non = 0
    local tot = 0
    local idx = 0
    -- for each training example
    for i,v in pairs(txts) do
       local siglen = #v + 2*zp
       local vect = {}
       for z=1,zp do
	  table.insert(0)
       end
       for j,tok in pairs(v) do
	  local z, lnon = idxFor(w2v, tok)
	  non = non + lnon
	  tot = tot + 1.0
	  table.insert(vect, z)
       end
       for z=1,zp do
	  table.insert(0)
       end
       -- this creates a batch of size 1 for now
       local tab = {vect}
       xt[i] = torch.LongTensor(tab)
       if options.batch2ndDim then
	  xt[i] = xt[i]:t()
       end

    end

    print('Sparsity ' .. (non/tot))
    ts.f2i = f2i
    ts.y = yt
    ts.x = xt
    ts.txts = txts
    return ts

end

-- Try to get an index using multiple approaches
function tryGetWordIdx(w2v, word)
   z = w2v.vocab[word]
   if z == nil then
      return w2v.vocab[word:lower()]
   else
      return z
   end

end

-- Get the index for a word using multiple approaches
function idxFor(w2v, tok)
   local non = 0

   z = tryGetWordIdx(w2v, tok)

   if z == nil then

      local alt = newWord(tok) -- w2v:lookup(tok, true)
      z = tryGetWordIdx(w2v, alt)

      if z == nil then
	 non = non + 1.0
	 z = 0
      end
   end
   return z, non
end

--[[
  Read in CONLL data and produce an vector for each word.
  This assumes we are not fine-tuning
--]]

function conllSentsToVectors(file, w2v, zp, f2i, w2cv)
   
    local tsfile = io.open(file, 'r')
    local linenum = 1

    local lbl = {}
    local lbls = {}
    local ts = {}
    local txt = {}
    local txts = {}
    local xt = {}
    local yt = {}
    local x = {}
    local y = {}
    local idx = 0
    local wsz = w2v.dsz
    local csz = w2cv and w2cv.dsz or 0
    print('Word vector sz ' .. wsz)
    print('Char vector sz ' .. csz)
    for line in tsfile:lines() do
       local state = line:split('%s+')
       if #state == 0 then
	  -- time to return the set
	  
	  table.insert(txts, txt)
	  table.insert(lbls, lbl)
	  table.insert(yt, torch.Tensor(y))
	  txt = {}
	  lbl = {}
	  y = {}
       else
	  local label = state[#state]
	  if not f2i[label] then
	     idx = idx + 1
	     f2i[label] = idx
	     print('Label ' .. label)
	  end

	  local word = state[1]
	  table.insert(txt, word)
	  table.insert(lbl, label)
	  table.insert(y, f2i[label])
       end
    end

    local non = 0
    local tot = 0
    local idx = 0
    -- for each training example
    for i,v in pairs(txts) do
       local siglen = #v + 2*zp
       local vec = torch.zeros(1, siglen, wsz + csz)
       for j,tok in pairs(v) do
	  local z, lnon = vecFor(w2v, tok)
	  non = non + lnon
	  tot = tot + 1.0
	  local cat = 0
	  if w2cv then
	     local q = word2chvec(w2cv, tok)
	     cat = torch.cat(z, q)
	  else
	     cat = z
	  end
	  if hasNaN(z) then
	     print('word ' .. tok .. ' has nan in vec')
	  else
	     vec[{{},(j+zp)}] = cat
	  end
       end
       xt[i] = vec
    end

    print('Sparsity ' .. (non/tot))
    ts.f2i = f2i
    ts.y = yt
    ts.x = xt
    ts.txts = txts
    return ts

end
