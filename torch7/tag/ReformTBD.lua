local ReformTBD, parent = torch.class('ReformTBD', 'nn.Module')

function ReformTBD:__init(T, D)
   parent.__init(self)
   self.T = T
   self.D = D
end

function ReformTBD:updateOutput(input)

   if not input:isContiguous() then
      self._input = self._input or input.new()
      self._input:resizeAs(input)
      self._input:copy(input)
      input = self._input
   end

   self.B = input:size(1) / self.T
   self.output:view(input, torch.LongStorage({self.T, self.B, self.D}))
   return self.output
end

function ReformTBD:updateGradInput(input, gradOutput)
   if not gradOutput:isContiguous() then
      self._gradOutput = self._gradOutput or gradOutput.new()
      self._gradOutput:resizeAs(gradOutput)
      self._gradOutput:copy(gradOutput)
      gradOutput = self._gradOutput
   end

   self.gradInput:viewAs(gradOutput, input)
   return self.gradInput
end

function ReformTBD:__tostring__()
  return torch.type(self) .. '(' ..
      table.concat(self.size:totable(), 'x') .. ')'
end

function ReformTBD:clearState()
   nn.utils.clear(self, '_input', '_gradOutput')
   return parent.clearState(self)
end

