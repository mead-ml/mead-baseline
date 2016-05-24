require 'nn'
require 'optim'
require 'xlua'

function createCrit(gpu)
   local crit = nn.HingeEmbeddingCriterion(1)
   return gpu and crit:cuda() or crit
end

function trainEpoch(crit, model, ts, optmeth, confusion, options)
    local xt = ts.x
    local yt = ts.y

    model:training()
    time = sys.clock()

    local shuffle = torch.randperm(#xt)
    w,dEdw = model:getParameters()
    for i=1,#xt do

       -- batch size is the first dimension
       local si = shuffle[i]
       local x = xt[si]
       if options.gpu then
	  x[1] = x[1]:cuda()
	  x[2] = x[2]:cuda()
       end
       local y = options.gpu and yt[si]:cuda() or yt[si]
       local thisBatchSz = y:size(1)

       local evalf = function(wt)

	  if wt ~= w then
	     print('Warning, params diff')
	     w:copy(wt)
	  end
	  
	  dEdw:zero()

	  local pred = model:forward(x)
	  local err = crit:forward(pred, y)

	  for j = 1,thisBatchSz do
	     local yg = pred[j] > 0 and 1 or 0
	     confusion:add(yg + 1, y[j] + 1)
	  end
  
	  local grad = crit:backward(pred, y)
	  model:backward(x, grad)
	  return err, dEdw
       end
       
       optmeth(evalf, w, config)
       if options.afteroptim then
	  options.afteroptim()
       end

       xlua.progress(i, #xt)
       
    end
    time = sys.clock() - time
    print(confusion)
    print('Training error ' .. (1-confusion.totalValid))
    print("Time to learn epoch " .. time .. 's')

end

function test(crit, model, es, confusion, options)

    local xt = es.x
    local yt = es.y
    model:evaluate()
    time = sys.clock()

    for i=1,#xt do
       local x = xt[i]
       if options.gpu then
	  x[1] = x[1]:cuda()
	  x[2] = x[2]:cuda()
       end

        local y = options.gpu and yt[i]:cuda() or yt[i]
	local thisBatchSz = y:size(1)
	local pred = model:forward(x)
--	local err = crit:forward(pred, y)

	for j = 1,thisBatchSz do
	   local yg = pred[j] > 0 and 1 or 0
	   confusion:add(yg + 1, y[j] + 1)
	end
  
	xlua.progress(i, #xt)

    end

    time = sys.clock() - time
    print(confusion)
    print('Test error ' .. (1-confusion.totalValid))
    print("Time to test " .. time .. 's')

    return (1-confusion.totalValid)
end
