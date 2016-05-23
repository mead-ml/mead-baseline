
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

function buildVocab(idx, files)
    local vocab = {['<GO>']=1,
                   ['<EOS>']=1,
		   ['URL']=1,
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
	  local cols = line:split('\t')
	  local sent = cols[idx]
	  
	  local words = sent:split('%s+')
	  for j=1,#words do
	     local w = words[j]
	     -- Allow either case for backoff
	     vocab[w] = vocab[w] or 1
	     w = w:lower()
	     vocab[w] = vocab[w] or 1
	  end
       end
    end
    
    return vocab
end

function readLines(tsfile)
   local srcs = {}
   local dsts = {}
   for line in tsfile:lines() do
      local splits = line:split('\t')
      local src = splits[1]:split('%s+')
      local dst = splits[2]:split('%s+')
      if #src < 1 or #dst < 1 then
	 print(#src)
	 print(#dst)
      end
      table.insert(srcs, src)
      table.insert(dsts, dst)
   end
   
   return srcs, dsts
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

   local OOV = w2v.vocab['<PADDING>']
   z = tryGetWordIdx(w2v, tok)

   if z == nil then

      local alt = newWord(tok) -- w2v:lookup(tok, true)
      z = tryGetWordIdx(w2v, alt)

      if z == nil then
	 z = OOV
      end
   end
   return z
end

function sentsToIndices(file, embed1, embed2, options)

    local tsfile = io.open(file, 'r')
    local linenum = 1
    
    local wsz = embed1.dsz
    local batchsz = options.batchsz or 1
    local mxlen = options.mxlen or 40

    -- Pad is always at 1!
    local PAD = embed1.vocab['<PADDING>']
    local GO = embed2.vocab['<GO>']
--    print(embed2:lookup('<GO>'))
    local EOS = embed2.vocab['<EOS>']

    local ts =  {}
    local srcs = {}
    local dsts = {}
    local tgts = {}

    srcs, dsts = readLines(tsfile)

    local thisBatchSz = nil
    local srcl = nil
    local dstl = nil
    local tgtl = nil
    local srct = {}
    local dstt = {}
    local tgtt = {}

    local idx = 0
    local b = 0
    for i,src in pairs(srcs) do

       local dst = dsts[i]
       local offset = (i - 1) % batchsz

       if offset == 0 then
	  if b > 0 then
	     srct[b] = srcl
	     dstt[b] = dstl
	     tgtt[b] = tgtl
	  end
	  b = b + 1
	  thisBatchSz = math.min(batchsz, #srcs - i + 1)
	  srcl = torch.LongTensor(thisBatchSz, mxlen):fill(PAD)
	  dstl = torch.LongTensor(thisBatchSz, mxlen + 1):fill(PAD)
	  tgtl = torch.LongTensor(thisBatchSz, mxlen + 1):fill(0)
       end

       end1 = math.min(#src, mxlen)
       end2 = math.min(#dst, mxlen)

       -- Usually less than mxlen we hope
       mxsiglen = math.max(end1, end2)
       dstl[{offset+1, 1}] = GO

       
       for j=1,mxsiglen do
	  local idx1 = j > end1 and PAD or idxFor(embed1, src[j])
	  local idx2 = j > end2 and PAD or idxFor(embed2, dst[j])

	  -- First signal is reversed and starting at end, left padding
	  srcl[{offset+1, mxlen - j + 1}] = idx1
	  -- Second signal is not reversed, follows <go> and ends on <eos>
	  dstl[{offset+1, j + 1}] = idx2
	  tgtl[{offset+1, j}] = idx2 == PAD and 0 or idx2
	  
       end
       tgtl[{offset+1, end2 + 1}] = EOS
    
    end


    if thisBatchSz > 0 then
       srct[b] = srcl
       dstt[b] = dstl
       tgtt[b] = tgtl
    end

    ts.dst = dstt
    ts.src = srct
    ts.tgt = tgtt
    return ts

end
