require 'nn'
require 'optim'
require 'xlua'

function createDistanceCrit(gpu)
   local crit = nn.HingeEmbeddingCriterion()
   return gpu and crit:cuda() or crit
end

function trainEpoch(crit, model, ts, optmeth, options)

    model:training()
    time = sys.clock()

    local sz = ts:size()
    local shuffle = torch.randperm(sz)
    w,dEdw = model:getParameters()

    local epochErr = 0

    for i=1,sz do

       -- batch size is the first dimension
       local si = shuffle[i]
       local batch = ts:get(si)
       local x = batch.x
       if options.gpu then
	  x[1] = x[1]:cuda()
	  x[2] = x[2]:cuda()
       end
       local y = options.gpu and batch.y:cuda() or batch.y
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

       xlua.progress(i, sz)
       
    end
    time = sys.clock() - time
    local avgEpochErr = epochErr / sz
    print('Train avg loss ' .. avgEpochErr)
    print("Time to learn epoch " .. time .. 's')

end

function test(crit, model, rlut, es, options)

    local show = options.show and options.show or 20
    model:evaluate()
    time = sys.clock()
    
    local epochErr = 0
    local i, j
    local ranks = {}
    local sz = es:size()
    for i=1,sz do

       local batch = es:get(i)
       local x = batch.x
       if options.gpu then
	  x[1] = x[1]:cuda()
	  x[2] = x[2]:cuda()
       end

        local y = options.gpu and batch.y:cuda() or batch.y
	local thisBatchSz = y:size(1)
	local pred = model:forward(x)
	local err = crit:forward(pred, y)
	for j = 1,thisBatchSz do
	   local score = pred[j]
	   table.insert(ranks, {i, j, score})
	end
	epochErr = epochErr + err
	xlua.progress(i, sz)

    end

    time = sys.clock() - time

    table.sort(ranks, function(a,b) return a[3]>b[3] end)

    j = 0
    for _,v in pairs(ranks) do
       if j > show then break end
       j = j + 1
       local idx = v[1]

       local batch = es:get(idx)
       local bidx = v[2]
       local score = string.format('%.2f', v[3])
       print('===========================================================')
       
       print('[' .. score .. '] (' .. batch.y[bidx] .. ')')
       s1 = lookupSent(rlut, batch.x[1][bidx])
       s2 = lookupSent(rlut, batch.x[2][bidx])
       print(s1)
       print(s2)
       print('-----------------------------------------------------------')
    end

    local avgEpochErr = epochErr / sz
    print('Test avg loss ' .. avgEpochErr)
    print("Time to run test " .. time .. 's')

    return avgEpochErr
end

