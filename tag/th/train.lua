require 'nn'
require 'optim'
require 'xlua'

function createTaggerCrit(gpu)
   local crit = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1))
   return gpu and crit:cuda() or crit
end

function testTagger(model, es, crit, confusion, options)

    local xt = es.x
    local yt = es.y

    model:evaluate()
    time = sys.clock()

    local epochErr = 0
    for i=1,#xt do
 	local x = options.gpu and xt[i]:cuda() or xt[i]
        local y = options.gpu and yt[i]:cuda() or yt[i]
	x = x:transpose(1, 2)
	y = y:transpose(1, 2)

	local pred = model:forward(x)

	local err = crit:forward(pred, y)
	epochErr = epochErr + err
	
	local thisBatchSz = x:size(2)
	local seqlen = x:size(1)

	-- Turn this from a table back into a tensor
	pred = torch.cat(pred, 1)
	local outcomes = pred:size(2)

	pred = pred:reshape(seqlen, thisBatchSz, outcomes)
	pred = pred:transpose(1, 2)
	x = x:transpose(1, 2)
	y = y:transpose(1, 2)

	for b=1,thisBatchSz do
	   local seq = pred[b]
	   local cy = y[b]:int()
	   _, path = seq:max(2)
	   local cpath = path:int():reshape(seqlen)
	   for j=1,seqlen do
	      local guessj = cpath[j]
	      local truthj = cy[j]
	      if truthj ~= 0 then
		 confusion:add(guessj, truthj)
	      end
	   end
	end
	
	xlua.progress(i, #xt)
    end
    
    print(confusion)
    local err = (1-confusion.totalValid)
    local avgEpochErr = epochErr / #xt
    time = sys.clock() - time
    print('Test avg loss ' .. avgEpochErr)
    print('Test accuracy (error) ' .. err)
    print("Time to test " .. time .. 's')
    return err
end

function trainTaggerEpoch(crit, model, ts, optmeth, options)
    local xt = ts.x
    local yt = ts.y
    model:training()
    local time = sys.clock()

    local shuffle = torch.randperm(#xt)
    w,dEdw = model:getParameters()
    local epochErr = 0
    for i=1,#xt do
       local si = shuffle[i]
	    
       local evalf = function(wt)
	  if wt ~= w then
	     print('Warning, params diff')
	     w:copy(wt)
	  end
	  
	  dEdw:zero()
	  local x = options.gpu and xt[si]:cuda() or xt[si]
	  local y = options.gpu and yt[si]:cuda() or yt[si]

	  x = x:transpose(1, 2)
	  y = y:transpose(1, 2)

	  local pred = model:forward(x)
	  local err = crit:forward(pred, y)
	  epochErr = epochErr + err

	  local grad = crit:backward(pred, y)
	  model:backward(x, grad)

	  if options.clip and options.clip > 0 then
	     dEdw:clamp(-options.clip, options.clip)
	  end
	  
	  return err,dEdw
       end

       optmeth(evalf, w, state)
       if options.afteroptim then
	  options.afteroptim()
       end

       xlua.progress(i, #xt)

    end
    time = sys.clock() - time
    local avgEpochErr = epochErr / #xt
    print('Train avg loss ' .. avgEpochErr)
    print("Time to learn epoch " .. time .. 's')

end
