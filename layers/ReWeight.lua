local ReWeight, Parent = torch.class('nn.ReWeight', 'nn.Module')

local eps = 1e-12

function ReWeight:__init()
  Parent.__init(self)
  self.ObjectnessScore = nil
  self.ReWeightPattern = SETTINGS.CO_TRAINING_PATTERN
end

function ReWeight:updateOutput(input)

  local nDim = input[1][1]:size(2)
  local nCls = input[2]:size(2)-1

  if not epoch_id then
    epoch_id = SETTINGS.test_epoch_num
  end

  if SETTINGS.annealFromEpoch3 then
    epoch_id = math.max(1, epoch_id-1)
  end

  local aneal_rate = nil

  if self.ReWeightPattern == 'None' then
    aneal_rate = 0
  end
  if self.ReWeightPattern == '0.5' then
    aneal_rate = 0.5
  end
  if self.ReWeightPattern == 'Anneal0.63' then
    aneal_rate = (epoch_id-1)/SETTINGS.NUM_EPOCHS/1.5
  end
  if self.ReWeightPattern == 'Anneal0.5' then
    aneal_rate = (epoch_id-1)/SETTINGS.NUM_EPOCHS/2
  end
  assert(aneal_rate, 'ReWeight PATTERN not supperted!')
  
  self.ObjectnessScore = input[#input][{{},{1,nCls}}]:max(2):repeatTensor(1,nDim)
  self.ObjectnessScore = (1-aneal_rate)+aneal_rate*self.ObjectnessScore
  --self.ObjectnessScore = torch.CudaTensor(#input[1]):fill(1)

  self.output = {}
  self.output[1] = {}
  self.output[1][1] = torch.cmul(input[1][1],self.ObjectnessScore)
  self.output[1][2] = torch.cmul(input[1][2],self.ObjectnessScore)
  self.output[1][3] = torch.cmul(input[1][3],self.ObjectnessScore)
  for i=1, #input-1 do
    self.output[i+1] = input[i+1]
  end

  return self.output
end

function ReWeight:updateGradInput(input, gradOutput)
  self.gradInput = {}
  self.gradInput[1] = {}
  self.gradInput[1][1] = torch.cmul(gradOutput[1][1],self.ObjectnessScore)
  self.gradInput[1][2] = torch.cmul(gradOutput[1][2],self.ObjectnessScore)
  self.gradInput[1][3] = torch.cmul(gradOutput[1][3],self.ObjectnessScore)

  for i=1, #gradOutput-1 do
    self.gradInput[i+1] = gradOutput[i+1]
  end

  return self.gradInput
end