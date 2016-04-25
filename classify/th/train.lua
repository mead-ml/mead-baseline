require 'nn'
require 'optim'
require 'xlua'

function createCrit(gpu)
   local crit = nn.ClassNLLCriterion()
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
       local x = options.gpu and xt[si]:cuda() or xt[si]
       local y = options.gpu and yt[si]:cuda() or yt[si]
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
  
	  local dEdy = crit:backward(pred, y)
	  model:backward(x, dEdy)
	  return err, dEdw
       end
       
       optmeth(evalf, w, state)
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

function test(model, es, confusion, options)

    local xt = es.x
    local yt = es.y
    model:evaluate()
    time = sys.clock()

    for i=1,#xt do
 	local x = options.gpu and xt[i]:cuda() or xt[i]
        local y = options.gpu and yt[i]:cuda() or yt[i]
	local thisBatchSz = x:size(1)
	local pred = model:forward(x)

	for j = 1,thisBatchSz do
	   confusion:add(pred[j], y[j])
	end
  
	xlua.progress(i, #xt)

    end
    time = sys.clock() - time
    print(confusion)
    print('Test error ' .. (1-confusion.totalValid))
    print("Time to test " .. time .. 's')
    return (1-confusion.totalValid)
end
