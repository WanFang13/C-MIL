fc8r = nn.Linear(base_model.fc_layers_output_size, numClasses):named('fc8r')

selector_module = nn.ConcatTable():
	add(nn.Sequential():
		add(nn.SelectTable(1)):
		add(nn.Linear(base_model.fc_layers_output_size, numClasses):named('fc8c')):
		add(RoiReshaper:RestoreShape()):
    add(nn.Squeeze(1)):
		named('output_fc8c')
	):
	add(nn.Sequential():
		add(nn.ConcatTable():
			add(nn.Sequential():
				add(nn.SelectTable(2)):
				add(share_weight_bias(fc8r)):
				named('output_fc8d_origring')
			):
			add(nn.Sequential():
				add(nn.SelectTable(3)):
				add(share_weight_bias(fc8r)):
				add(nn.MulConstant(-1)):
				named('output_fc8d_context')
			)
		):
		add(nn.CAddTable()):
		add(RoiReshaper:RestoreShape(4))
	)


model = nn.Sequential():
	add(nn.ParallelTable():
		add(base_model.conv_layers):
		add(RoiReshaper:StoreShape())
	):
	add(nn.ConcatTable():
		add(branch_transform_rois_share_fc_layers(base_model, BoxOriginal)):
		add(branch_transform_rois_share_fc_layers(base_model, BoxOriginal_ring)):
		add(branch_transform_rois_share_fc_layers(base_model, ContextRegion))
	):
	add(nn.ConcatTable():
    add(nn.Identity()):
    add(nn.Sequential():
			add(nn.SelectTable(1)):
      add(nn.Linear(base_model.fc_layers_output_size, numClasses+1):named('det_fc81')):
      add(cudnn.SpatialSoftMax())
    ):
    add(nn.Sequential():
			add(nn.SelectTable(1)):
      add(nn.Linear(base_model.fc_layers_output_size, numClasses+1):named('det_fc82')):
      add(cudnn.SpatialSoftMax())
    )
  ):
  add(nn.ReWeight()):
  add(nn.ParallelTable():
    add(nn.Sequential():
  		add(selector_module):
      add(nn.ContinuationSubset())
    ):
    add(nn.Identity()):
    add(nn.Identity())
  ):
  add(nn.ContinuationDetector())

criterion = nn.ParallelCriterion():
  add(HingeCriterion():setFactor(1 / numClasses), 1):
  add(nn.ClassNLLCriterion(), SETTINGS.DetRate):
  add(nn.ClassNLLCriterion(), SETTINGS.DetRate)

optimState          = {
  learningRate = SETTINGS.LearningRate,
  momentum     = 0.9, 
  weightDecay  = 5e-4
}
optimState_annealed = {
  learningRate = SETTINGS.LearningRateAneal,
  momentum     = 0.9, 
  weightDecay  = 5e-4, 
  epoch        = SETTINGS.AnnealEpoch
}