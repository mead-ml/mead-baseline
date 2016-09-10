require 'nn'
require 'optim'
require 'xlua'
require 'utils'

-- Decoder beam
SAMPLE_PRUNE_INIT = 5

function trainSeq2SeqEpoch(crit, model, ts, optmeth, options)

    local enc = model:get(1)
    local dec = model:get(2)
    enc:training()
    dec:training()
    local sz = ts:size()
    local time = sys.clock()

    local shuffle = torch.randperm(sz)
    w,dEdw = getModelParameters(model)

    local epochErr = 0
    for i=1,sz do
       local si = shuffle[i]
	    
       local evalf = function(wt)
	  if wt ~= w then
	     print('Warning, params diff')
	     w:copy(wt)
	  end
	  
	  dEdw:zero()

	  local batch = ts:get(si)
	  
	  local src = batch.src
	  local dst = batch.dst
	  local tgt = batch.tgt

	  if options.gpu then
	     src = src:cuda()
	     dst = dst:cuda()
	     tgt = tgt:cuda()
	  end

	  src = src:transpose(1, 2)
	  dst = dst:transpose(1, 2)
	  tgt = tgt:transpose(1, 2)
	  tgtTable = tab1st(tgt)
	  local predSrc = enc:forward(src)
	  local srclen = src:size(1)
	  forwardConnect(model, srclen, options.layers)
	  
	  local predDst = dec:forward(dst)
	  local err = crit:forward(predDst, tgtTable)
	  epochErr = epochErr + err

	  local grad = crit:backward(predDst, tgtTable)
	  dec:backward(dst, grad)

	  backwardConnect(model, options.layers)
	  
	  enc:backward(src, predSrc:zero())

	  forgetModelState(model)

	  if options.clip and options.clip > 0 then
	     dEdw:clamp(-options.clip, options.clip)
	  end
	  
	  return err,dEdw
       end

       optmeth(evalf, w, state)
       if options.afteroptim then
	  options.afteroptim()
       end
       xlua.progress(i, sz)

    end
    time = sys.clock() - time
    local avgEpochErr = epochErr / sz
    print('Train avg loss ' .. avgEpochErr)
    print("Time to learn epoch " .. time .. 's')

end

function decodeStep(model, srcIn, predSent, j, sample, layers)
   forgetModelState(model)

   local enc = model:get(1)
   local predSrc = enc:forward(srcIn)

   forwardConnect(model, srcIn:size(1), layers)

   local dec = model:get(2)
   local predT = torch.LongTensor(predSent):reshape(j, 1)
   local predDst = dec:forward(predT)[j]
   local word = nil
   if sample then

      local probs = predDst:squeeze():exp()
      -- Get the topk
      local beamsz = math.max(SAMPLE_PRUNE_INIT - j, 1)
      local best, ids = probs:topk(beamsz, 1, true, true)
      -- log soft max, exponentiate to get probs
      probs:zero()
      probs:indexCopy(1, ids, best)
      probs:div(torch.sum(probs))
      word = torch.multinomial(probs, 1):squeeze()
      --print('word ' .. word)
   else
      local _, ids = predDst:max(2)
      word = ids:squeeze()
   end
   return word
end

function decode(model, srcIn, sample, GO, EOS, layers)

   local predSent = {GO}
   for j = 1,100 do
      local word = decodeStep(model, srcIn, predSent, j, sample, layers)
      if word == EOS then
	 break
      end
      table.insert(predSent, word)

   end
   return predSent
end

-- To really do this during decoding, you want a beam search over multiple words
-- to maximize probability of the sentence.  This is fairly straightforward,
-- but slow and not really required during training.
-- Here, for speed and simplicity, just show some examples by pruning 
-- vocab down to most likely words and
-- doing a single independent multinomial draw greedily.

function showBatch(model, ts, rlut1, rlut2, embed2, opt)
   -- When a batch comes in, it will be BxT
   -- so to print a batch input, walk the batch and 

   local GO = embed2.vocab['<GO>']
   local EOS = embed2.vocab['<EOS>']

   local sz = ts:size()

   gen = torch.Generator()
   local rnum = torch.random(gen, 1, sz)
   local batch = ts:get(rnum)
   local src = batch.src
   local dst = batch.dst
   local tgt = batch.tgt

   local batchsz = src:size(1)
   local num = math.min(batchsz, opt.showex)

   local method = opt.sample and 'Sampling' or 'Showing best'
   print(method .. ' sentences from batch #' .. rnum)
   -- Run whole batch through

   for i=1,num do

      local sent = indices2sent(rlut1, src[i], true)
      print('========================================================================')
      print('[OP]', sent)
      sent = indices2sent(rlut2, tgt[i], false)
      print('[Actual]', sent)

      local srclen = src[i]:size(1)
      local srcIn = src[i]:reshape(srclen, 1)
      srcIn = opt.gpu and srcIn:cuda() or srcIn
      
      predSent = decode(model, srcIn, opt.sample, GO, EOS, opt.layers)
      sent = indices2sent(rlut2, torch.LongTensor(predSent))
      print('Guess: ', sent)
      print('------------------------------------------------------------------------')

   end
   forgetModelState(model)

end

function testSeq2Seq(model, ts, crit, options)

    local enc = model:get(1)
    local dec = model:get(2)
    enc:evaluate()
    dec:evaluate()
    local sz = ts:size()
    local time = sys.clock()

    local epochErr = 0
    for i=1,sz do

       local batch = ts:get(i)
       local src = batch.src
       local dst = batch.dst
       local tgt = batch.tgt

       if options.gpu then
	  src = src:cuda()
	  dst = dst:cuda()
	  tgt = tgt:cuda()
       end
       src = src:transpose(1, 2)
       dst = dst:transpose(1, 2)
       tgt = tgt:transpose(1, 2)
       local tgtTable = tab1st(tgt)
	  
       local predSrc = enc:forward(src)
       local srclen = src:size(1)
       forwardConnect(model, srclen, options.layers)
       
       local predDst = dec:forward(dst)
       
       local err = crit:forward(predDst, tgtTable)
       epochErr = epochErr + err
       xlua.progress(i, sz)

       forgetModelState(model)

    end
    time = sys.clock() - time
    local avgEpochErr = epochErr / sz
    print('Test avg loss ' .. avgEpochErr)
    print("Time to run test " .. time .. 's')

    return avgEpochErr
end
