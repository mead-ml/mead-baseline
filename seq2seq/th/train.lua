require 'nn'
require 'optim'
require 'xlua'
require 'seq2sequtils'

function trainSeq2SeqEpoch(crit, model, ts, optmeth, options)
    local srcs = ts.src
    local dsts = ts.dst
    local tgts = ts.tgt

    local enc = model:get(1)
    local dec = model:get(2)
    enc:training()
    dec:training()

    local time = sys.clock()

    local shuffle = torch.randperm(#tgts)
    w,dEdw = getModelParameters(model)

    local epochErr = 0
    for i=1,#tgts do
       local si = shuffle[i]
	    
       local evalf = function(wt)
	  if wt ~= w then
	     print('Warning, params diff')
	     w:copy(wt)
	  end
	  
	  dEdw:zero()

	  local src = options.gpu and srcs[si]:cuda() or srcs[si]
	  local dst = options.gpu and dsts[si]:cuda() or dsts[si]
	  local tgt = options.gpu and tgts[si]:cuda() or tgts[si]

	  src = src:transpose(1, 2)
	  dst = dst:transpose(1, 2)
	  tgt = tgt:transpose(1, 2)

	  local predSrc = enc:forward(src)
	  local srclen = src:size(1)
	  forwardConnect(model, srclen)
	  
	  local predDst = dec:forward(dst)
	  local err = crit:forward(predDst, tgt)
	  epochErr = epochErr + err

	  local grad = crit:backward(predDst, tgt)
	  dec:backward(dst, grad)

	  backwardConnect(model)
	  
	  enc:backward(src, predSrc:zero())

	  if options.clip and options.clip > 0 then
	     dEdw:clamp(-options.clip, options.clip)
	  end
	  
	  return err,dEdw
       end

       optmeth(evalf, w, state)
       if options.afteroptim then
	  options.afteroptim()
       end

       xlua.progress(i, #tgts)

    end
    time = sys.clock() - time
    local avgEpochErr = epochErr / #tgts
    print('Train avg loss ' .. avgEpochErr)
    print("Time to learn epoch " .. time .. 's')

end


function decodeStep(model, srcIn, predSent, j)
   forgetModelState(model)

   local enc = model:get(1)
   local predSrc = enc:forward(srcIn)

   forwardConnect(model, srcIn:size(1))

   local dec = model:get(2)
   local predT = torch.LongTensor(predSent):reshape(j, 1)
   local predDst = dec:forward(predT)[j]
   -- TODO: beam, sample
   local _, ids = predDst:topk(1, 2, true, true)
   return ids[1][1]
end

function decode(model, srcIn, GO, EOS)

   local predSent = {GO}
   for j = 1,100 do
      local word = decodeStep(model, srcIn, predSent, j)
      if word == EOS then
	 break
      end
      table.insert(predSent, word)

   end
   return predSent
end

function showBatch(model, ts, rlut1, rlut2, embed2, gpu)
   -- When a batch comes in, it will be BxT
   -- so to print a batch input, walk the batch and 

   local GO = embed2.vocab['<GO>']
   local EOS = embed2.vocab['<EOS>']
   local srcs = ts.src
   local tgts = ts.tgt
   local dsts = ts.dst


   gen = torch.Generator()
   local rnum = torch.random(gen, 1, #srcs)
   local src = srcs[rnum]
   local dst = dsts[rnum]
   local tgt = tgts[rnum]

   local batchsz = src:size(1)

   print('Showing examples from batch #' .. rnum)
      
   -- Run whole batch through

   for i=1,batchsz do

      local sent = lookupSent(rlut1, src[i], true)
      print('========================================================================')
      print('[OP]', sent)
      sent = lookupSent(rlut2, tgt[i], false)
      print('[Actual]', sent)

      local srclen = src[i]:size(1)
      local srcIn = src[i]:reshape(srclen, 1)
      srcIn = gpu and srcIn:cuda() or srcIn
      
      predSent = decode(model, srcIn, GO, EOS)
      sent = lookupSent(rlut2, torch.LongTensor(predSent))
      print('Guess: ', sent)
      print('------------------------------------------------------------------------')

   end
   forgetModelState(model)

end

function testSeq2Seq(model, ts, crit, options)
    local srcs = ts.src
    local dsts = ts.dst
    local tgts = ts.tgt

    local enc = model:get(1)
    local dec = model:get(2)
    enc:evaluate()
    dec:evaluate()

    local time = sys.clock()

    local shuffle = torch.randperm(#tgts)
    local epochErr = 0
    for i=1,#tgts do
       local si = shuffle[i]

       local src = options.gpu and srcs[si]:cuda() or srcs[si]
       local dst = options.gpu and dsts[si]:cuda() or dsts[si]
       local tgt = options.gpu and tgts[si]:cuda() or tgts[si]
       
       src = src:transpose(1, 2)
       dst = dst:transpose(1, 2)
       tgt = tgt:transpose(1, 2)
	  
       local predSrc = enc:forward(src)
       
       local srclen = src:size(1)
       forwardConnect(model, srclen)
       
       local predDst = dec:forward(dst)
       
       local err = crit:forward(predDst, tgt)
       epochErr = epochErr + err
       xlua.progress(i, #tgts)

       forgetModelState(model)

    end
    time = sys.clock() - time
    local avgEpochErr = epochErr / #tgts
    print('Test avg loss ' .. avgEpochErr)
    print("Time to run test " .. time .. 's')

    return avgEpochErr
end
