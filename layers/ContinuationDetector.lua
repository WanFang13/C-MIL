local ContinuationDetector, Parent = torch.class('nn.ContinuationDetector', 'nn.Module')

local eps = 1e-12

function GetOverlap(boxes_score_im, rois, gt_image_label)
  local _, max_inds = boxes_score_im:max(1)
  local gt_inds  = max_inds[1][torch.eq(gt_image_label[1],1)]:long()
  local gt_rois  = rois:index(1, gt_inds):cuda()
  local overlaps = torch.CudaTensor(rois:size(1), gt_rois:size(1))
  overlaps.salc.ME_ComputeOverlap(
    self,
    rois,
    gt_rois,
    overlaps,
    gt_rois:size(1),
    rois:size(1)
  )
  return overlaps
end

function GetLabels(max_overlap, max_inds, gt_image_label)
  local gt_mask = torch.eq(gt_image_label[1],1)
  local gt_inds = torch.range(1,20):cuda()[gt_mask]:view(-1,1)
  local labels = gt_inds:index(1, max_inds:view(-1):long())

  local neg_inds = torch.lt(max_overlap, 0.5)
  labels[neg_inds] = 21

  return labels
end

function GetBoxesInds(max_overlap, nPosRoi, nNegRoi)
  local nRoi = max_overlap:size(1)
  local neg_mask1 = torch.ge(max_overlap:view(-1), 0.1)
  local neg_mask2 = torch.lt(max_overlap:view(-1), 0.4)
--  local neg_mask2 = torch.lt(max_overlap:view(-1), 0.4*SETTINGS.lambda)
--  local neg_mask2 = torch.lt(max_overlap:view(-1), 0.5*SETTINGS.lambda)
  neg_mask = torch.cmul(neg_mask1,neg_mask2)
  local pos_mask = torch.ge(max_overlap:view(-1), 0.6)
--  local pos_mask = torch.ge(max_overlap:view(-1), 1-0.4*SETTINGS.lambda)
--  local pos_mask = torch.ge(max_overlap:view(-1), 1-0.5*SETTINGS.lambda)

  local neg_inds = torch.range(1,nRoi):cuda()[neg_mask]
  local pos_inds = torch.range(1,nRoi):cuda()[pos_mask]

  if neg_mask:sum() == 0 then
    neg_mask = torch.lt(max_overlap:view(-1), 0.4)
    neg_inds = torch.range(1,nRoi):cuda()[neg_mask]
  end
  nNegRoi = math.min(neg_mask:sum(), nNegRoi)
  nPosRoi = math.min(pos_mask:sum(), nPosRoi)

  NegRandperm = torch.randperm(neg_mask:sum()):long()
  PosRandperm = torch.randperm(pos_mask:sum()):long()

  neg_inds = neg_inds:index(1, NegRandperm[{{1,nNegRoi}}])
  pos_inds = pos_inds:index(1, PosRandperm[{{1,nPosRoi}}])

  return torch.cat(neg_inds,pos_inds,1):long()

end

function ContinuationDetector:__init()
	Parent.__init(self)
	self.nPosRoi  = 32
  self.nNegRoi  = 96
  self.inds             = {}
  self.boxes_score_det  = {}

  self.LogLayer = nn.Log():cuda()
end

function ContinuationDetector:updateOutput(input)
  local im_score        = input[1]
  local boxes_score_im  = ScorePred
  local rois            = scale0_rois[1]:cuda()

  self.inds             = {}
  self.boxes_score_det  = {}
  local boxes_score_det = {}
  boxes_score_det[1] = input[2]
  boxes_score_det[2] = input[3]

  if self.train then
    --init
    local gt_image_label = batch_labels_gpu
    batch_box_labels_gpu = {}

    local overlaps = GetOverlap(boxes_score_im, rois, gt_image_label)
    --get labels and max_overlaps
    local max_overlap, max_inds = overlaps:max(2)
    local labels = GetLabels(max_overlap, max_inds, gt_image_label)
    --get boxes inds
    self.inds[1] = GetBoxesInds(max_overlap, self.nPosRoi, self.nNegRoi)
    --find gt boxes and their labels using im gt label
    local labels_out = labels:index(1,self.inds[1]):view(-1)
    self.boxes_score_det[1] = boxes_score_det[1]:index(1,self.inds[1])
    self.boxes_score_det[1] = self.boxes_score_det[1] + 1e-5
    batch_box_labels_gpu[1] = labels_out

    local overlaps = GetOverlap(boxes_score_det[1][{{},{1,20}}], rois, gt_image_label)
    --get labels and max_overlaps
    local max_overlap, max_inds = overlaps:max(2)
    local labels = GetLabels(max_overlap, max_inds, gt_image_label)
    --get boxes inds
    self.inds[2] = GetBoxesInds(max_overlap, self.nPosRoi, self.nNegRoi)
    --find gt boxes and their labels using im gt label
    local labels_out = labels:index(1,self.inds[2]):view(-1)
    self.boxes_score_det[2] = boxes_score_det[2]:index(1,self.inds[2])
    self.boxes_score_det[2] = self.boxes_score_det[2] + 1e-5
    batch_box_labels_gpu[2] = labels_out

    self.output = {}
    self.output[1] = im_score
    for i=1, #self.boxes_score_det do
      self.output[i+1] = torch.log(self.boxes_score_det[i])
    end
  else
    batch_box_labels_gpu = torch.ones(rois:size(1)):cuda()
    self.output = {}
    self.output[1] = im_score
    self.output[2] = ScoreOutput:view(-1,20) --it is in LocalMinEnrtopy updateOutput
    for i=1, #boxes_score_det do
      self.output[i+2] = boxes_score_det[i][{{},{1,20}}]
    end
    assert(self.output[2]:size(1) == self.output[3]:size(1))
  end

  return self.output
end

function ContinuationDetector:updateGradInput(input, gradOutput)
  local im_score        = input[1]
  local rois            = scale0_rois[1]:cuda() --nRoi * 5 dim

  local boxes_score_det = {}
  boxes_score_det[1] = input[2]
  boxes_score_det[2] = input[3]

  self.gradInput = {}
  self.gradInput[1] = gradOutput[1]

  for i=1,#input-1 do
    self.gradInput[i+1] = torch.CudaTensor(#input[i+1]):fill(0)
    self.gradInput[i+1].salc.ME_BoxGradBackward(
      self,
      self.gradInput[i+1],
      nn.Log():cuda():backward(self.boxes_score_det[i], gradOutput[i+1]),
      self.inds[i]:cuda()-1, --inds start from 0 in C
      self.inds[i]:size(1),
      21
    )
  end

  return self.gradInput
end
