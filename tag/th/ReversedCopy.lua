local ReversedCopy, parent = torch.class('nn.ReversedCopy', 'nn.Module')

-- This is used for BLSTM using torch-rnn
function ReversedCopy:__init()
   parent.__init(self)
   self.gradInput = torch.getmetatable(torch.Tensor.__typename).new()
   self.output = torch.getmetatable(torch.Tensor.__typename).new()
end

function ReversedCopy:updateOutput(input)
   self.output:resize(input:size())
   
   -- Along the embedding
   local B = input:size(1)
   local T = input:size(2)
   local D = input:size(3)

   for b=1,B do
      for j=1,D do
	 local k = 1
	 for i=T,1,-1 do
	    self.output[{b,k,j}] = input[{b,i,j}]
	    k = k + 1
	 end
      end
   end
   return self.output
end

function ReversedCopy:updateGradInput(input, gradOutput)
   self.gradInput:resize(gradOutput:size())

   local B = input:size(1)
   local T = input:size(2)
   local H = input:size(3)
   for b=1,B do
      for j=1,H do
	 local k = 1
	 for i=T,1,-1 do
	    self.gradInput[{b,k,j}] = gradOutput[{b,i,j}]
	    k = k + 1
	 end
      end
   end
   return self.gradInput
end
