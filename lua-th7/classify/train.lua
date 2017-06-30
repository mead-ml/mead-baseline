require 'nn'
require 'optim'
require 'xlua'

function createCrit(gpu)
   local crit = nn.ClassNLLCriterion()
   return gpu and crit:cuda() or crit
end


function trainEpoch(crit, model, ts, optmeth, confusion, options)
--    local xt = ts.x
--    local yt = ts.y

    model:training()
    time = sys.clock()

    local sz = ts:size()
    local shuffle = torch.randperm(sz)
    w,dEdw = model:getParameters()
    for i=1,sz do

       -- batch size is the first dimension
       local si = shuffle[i]
       local txy = ts:get(si)
       local x = txy.x
       local y = txy.y
       if options.gpu then
	  x = x:cuda()
	  y = y:cuda()
       end
       local thisBatchSz = x:size(1)

       local evalf = function(wt)

	  if wt ~= w then
	     print('Warning, params diff')
	     w:copy(wt)
	  end
	  
	  dEdw:zero()

	  local pred = model:forward(x)
	  local err = crit:forward(pred, y)

	  for j = 1,thisBatchSz do
	     confusion:add(pred[j], y[j])
	  end
  
	  local grad = crit:backward(pred, y)
	  model:backward(x, grad)
	  return err, dEdw
       end
       
       optmeth(evalf, w, config)
       if options.afteroptim then
	  options.afteroptim()
       end

       xlua.progress(i, sz)
       
    end
    time = sys.clock() - time
    print(confusion)
    print('Training error ' .. (1-confusion.totalValid))
    print("Time to learn epoch " .. time .. 's')

end

function test(crit, model, es, confusion, options)

    
    model:evaluate()
    time = sys.clock()
    local sz = es:size()

    for i=1,sz do
        local exy = es:get(i) 
 	local x = exy.x
        local y = exy.y
	if options.gpu then
	   x = x:cuda()
	   y = y:cuda()
	end
	local thisBatchSz = x:size(1)
	local pred = model:forward(x)
--	local err = crit:forward(pred, y)

	for j = 1,thisBatchSz do
	   confusion:add(pred[j], y[j])
	end
  
	xlua.progress(i, sz)

    end

    time = sys.clock() - time
    print(confusion)
    print('Test error ' .. (1-confusion.totalValid))
    print("Time to test " .. time .. 's')

    return (1-confusion.totalValid)
end
