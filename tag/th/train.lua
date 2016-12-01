require 'nn'
require 'optim'
require 'xlua'

function createTaggerCrit(gpu)
   local crit = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1))
   return gpu and crit:cuda() or crit
end

function testTagger(phase, model, es, crit, i2f, options, txts)

    model:evaluate()
    confusion = optim.ConfusionMatrix(i2f)

    local file = nil
    if txts then
       file = io.open(options.output, 'w') or nil
    end

    time = sys.clock()
    local batchsz = options.batchsz
    local steps = math.floor(es:size()/batchsz)

    local epochErr = 0

    for i=1,steps do
        local xy = batch(es, i, batchsz)
 	local xch = options.gpu and xy.xch:cuda() or xy.xch
        local y = options.gpu and xy.y:cuda() or xy.y
	local id = xy.id
	local x_tbl = {}
	table.insert(x_tbl, xch)
	if options.embed ~= 'NONE' then
	   local x = options.gpu and xy.x:cuda() or xy.x
	   table.insert(x_tbl, x)
	end

	local pred = model:forward(x_tbl)
	local yt = tab1st(y:transpose(1, 2))

	local err = crit:forward(pred, yt)
	epochErr = epochErr + err

	local seqlen = xch:size(2)
	-- Turn this from a table back into a tensor
	pred = torch.cat(pred, 1)
	
	local outcomes = pred:size(2)

	pred = pred:reshape(seqlen, batchsz, outcomes)
	pred = pred:transpose(1, 2)
	
	for b=1,batchsz do
	   local seq = pred[b]
	   local cy = y[b]:int()
	   _, path = seq:max(2)
	   local cpath = path:int():reshape(seqlen)
	   for j=1,seqlen do
	      local guessj = cpath[j]
	      local truthj = cy[j]
	      if truthj ~= 0 then
		 confusion:add(guessj, truthj)
		 if file ~= nil then
		    file:write(txts[id[b]][j] .. ' ' .. i2f[truthj] .. ' ' .. i2f[guessj])
		    file:write('\n')
		 end
	      end
	   end
	   if file ~= nil then
	      file:write('\n')
	   end
	end
	
	xlua.progress(i, steps)
    end
    
    print(confusion)
    local err = (1-confusion.totalValid)
    local avgEpochErr = epochErr / steps
    time = sys.clock() - time

    if file ~= nil then
       file:close()
    end

    acc = 1 - err
    print(phase .. ' avg loss ' .. avgEpochErr)
    print(phase .. ' accuracy ' .. acc)
    print("elapsed " .. time .. 's')
    return avgEpochErr, acc
end

function trainTaggerEpoch(crit, model, ts, optmeth, options)
    model:training()
    local time = sys.clock()
    local batchsz = options.batchsz
    local steps = math.floor(ts:size()/batchsz)

    local shuffle = torch.randperm(steps)
    w,dEdw = model:getParameters()
    local epochErr = 0

    for i=1,steps do
       local si = shuffle[i]
	    
       local evalf = function(wt)
	  if wt ~= w then
	     print('Warning, params diff')
	     w:copy(wt)
	  end
	  
	  dEdw:zero()
	  local xy = batch(ts, si, batchsz)
	  local xch = options.gpu and xy.xch:cuda() or xy.xch
	  local y = options.gpu and xy.y:cuda() or xy.y
	  local x_tbl = {}
	  table.insert(x_tbl, xch)
	  if options.embed ~= 'NONE' then
	     local x = options.gpu and xy.x:cuda() or xy.x
	     table.insert(x_tbl, x)
	  end
	  local pred = model:forward(x_tbl)
	  yt = tab1st(y:transpose(1, 2))

	  
	  local err = crit:forward(pred, yt)
	  epochErr = epochErr + err
	  local grad = crit:backward(pred, yt)
	  model:backward(x_tbl, grad)

	  if options.clip and options.clip > 0 then
	     dEdw:clamp(-options.clip, options.clip)
	  end
	  
	  return err,dEdw
       end

       optmeth(evalf, w, state)
       if options.afteroptim then
	  options.afteroptim()
       end

       xlua.progress(i, steps)

    end

    time = sys.clock() - time
    local avgEpochErr = epochErr / steps
    print('Train avg loss ' .. avgEpochErr)
    print("Time to learn epoch " .. time .. 's')

end
