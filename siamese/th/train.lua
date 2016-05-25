require 'nn'
require 'optim'
require 'xlua'

function createDistanceCrit(gpu)
   local crit = nn.HingeEmbeddingCriterion()
   return gpu and crit:cuda() or crit
end

function trainEpoch(crit, model, ts, optmeth, options)
    local xt = ts.x
    local yt = ts.y

    model:training()
    time = sys.clock()

    local shuffle = torch.randperm(#xt)
    w,dEdw = model:getParameters()

    local epochErr = 0

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
	  epochErr = epochErr + err
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
    local avgEpochErr = epochErr / #xt
    print('Train avg loss ' .. avgEpochErr)
    print("Time to learn epoch " .. time .. 's')

end

function test(crit, model, rlut, es, options)

    local xt = es.x
    local yt = es.y
    local show = options.show and options.show or 20
    model:evaluate()
    time = sys.clock()
    
    local epochErr = 0
    local i, j
    local ranks = {}
    for i=1,#xt do
       local x = xt[i]
       if options.gpu then
	  x[1] = x[1]:cuda()
	  x[2] = x[2]:cuda()
       end

        local y = options.gpu and yt[i]:cuda() or yt[i]
	local thisBatchSz = y:size(1)
	local pred = model:forward(x)
	local err = crit:forward(pred, y)
	for j = 1,thisBatchSz do
	   local score = pred[j]
	   table.insert(ranks, {i, j, score})
	end
	epochErr = epochErr + err
	xlua.progress(i, #xt)

    end

    time = sys.clock() - time

    table.sort(ranks, function(a,b) return a[3]>b[3] end)

    j = 0
    for _,v in pairs(ranks) do
       if j > show then break end
       j = j + 1
       local idx = v[1]
       local bidx = v[2]
       local score = string.format('%.2f', v[3])
       print('===========================================================')
       
       print('[' .. score .. '] (' .. yt[idx][bidx] .. ')')
       s1 = lookupSent(rlut, xt[idx][1][bidx])
       s2 = lookupSent(rlut, xt[idx][2][bidx])
       print(s1)
       print(s2)
       print('-----------------------------------------------------------')
    end

    local avgEpochErr = epochErr / #xt
    print('Test avg loss ' .. avgEpochErr)
    print("Time to run test " .. time .. 's')

    return avgEpochErr
end

