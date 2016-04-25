require 'nn'
require 'optim'
require 'xlua'

---------------------------------------------------------
-- Create a cross-entropy loss function, using sequencer
-- if we are using ElementResearch's rnn
---------------------------------------------------------
function createTaggerCrit(gpu, usernnpkg)
   local crit = nil

   if usernnpkg then
      crit = nn.SequencerCriterion(nn.CrossEntropyCriterion())
   else
      crit = nn.CrossEntropyCriterion()
   end
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
	if options.batch2ndDim then
	   x = x:reshape(x:size(2), x:size(1), x:size(3))
	end

        local y = options.gpu and yt[i]:cuda() or yt[i]

	-- Safety check on crazy test data
	if y:dim() > 0 then
	   
	   local cy = y:int()	      
	   local seqlen = cy:size(1)

	   local pred = model:forward(x)
	   local err = crit:forward(pred, y)
	   epochErr = epochErr + err

	   -- ER rnn package likes a table
	   if type(pred) == 'table' then
	      pred = torch.cat(pred)
	   end

	   pred = pred:reshape(seqlen, confusion.nclasses)
	      
	   _, path = pred:max(2)
	   local cpath = path:int()

	   for j=1,seqlen do
	      local guessj = cpath[j][1]
	      local truthj = cy[j]
	      confusion:add(guessj, truthj)
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
	  -- ER rnn library prefers
	  if options.batch2ndDim then
	     x = x:reshape(x:size(2), x:size(1), x:size(3))
	  end

	  local y = options.gpu and yt[si]:cuda() or yt[si]
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
       xlua.progress(i, #xt)

    end
    time = sys.clock() - time
    local avgEpochErr = epochErr / #xt
    print('Train avg loss ' .. avgEpochErr)
    print("Time to learn epoch " .. time .. 's')

end
