local PrintIt, parent = torch.class('nn.PrintIt', 'nn.Module')

function PrintIt:__init(prefix, norm, idx)
   parent.__init(self)
   self.idx = idx
   self.norm = norm
   self.prefix = prefix or "PrintIt"
end

function PrintIt:updateOutput(input)
   self.output = input

   local z = self.idx == nil and input or input[self.idx]
   print(self.prefix..":input\n", self.norm and z:norm() or z)
   return self.output
end


function PrintIt:updateGradInput(input, gradOutput)

--   local z = self.idx == nil and gradOutput or gradOutput[self.idx]
--   print(self.prefix..":gradOutput\n", self.norm and z:norm() or z)
   self.gradInput = gradOutput
   return self.gradInput
end
