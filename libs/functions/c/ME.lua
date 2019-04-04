local ME, Parent = torch.class('nn.ME', 'nn.Module')

function ME:__init()
	Parent.__init(self)
end

function ME:updateOutput(input)
	self.output = 0
	return self.output
end

function ME:updateGradInput(input, gradOutput)
	self.gradInput = 0
	return self.gradInput
end