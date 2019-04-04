-- settings for path and models
dofile('settings.lua')

dofile('preprocess.lua')
dofile('opts.lua')
dofile('util.lua')
dofile('dataset.lua')
dofile('layers/util.lua')

require "lfs"

opts.PATHS.MODEL = opts.PATHS.CHECKPOINT_PATTERN:format(SETTINGS.test_epoch_num)
print("model load path:")
print(opts.PATHS.MODEL)

loaded = model_load(opts.PATHS.MODEL, opts)

meta = {
	opts = opts,
	training_meta = loaded.meta,
	example_loader_options = {
		evaluate = {
			numRoisPerImage = 8192,
			subset = SETTINGS.SUBSET_FOR_TESTING,
			hflips = true,
			numScales = opts.NUM_SCALES
		}
	}
}

batch_loader = ParallelBatchLoader(
	ExampleLoader(
		dataset, 
		base_model.normalization_params, 
		opts.IMAGE_SCALES, 
		meta.example_loader_options
	)
):setBatchSize({evaluate = 1})

print(meta)
assert(model):cuda()
assert(criterion):cuda()
collectgarbage()

tic_start = torch.tic()

batch_loader:evaluate()
model:evaluate()
scores, labels, rois, outputs, corlocs, log, corlocs_all = {},{},{},{},{},{},{}
for batchIdx = 1, batch_loader:getNumBatches() do
	tic = torch.tic()

	scale_batches = batch_loader:forward()[1]
	scale0_rois = scale_batches[1][2]
	scale_outputs, scale_scores, scale_costs = {}, {}, {}
	for i = 2, #scale_batches do
		batch_images, batch_rois, batch_labels = unpack(scale_batches[i])
		batch_images_gpu = torch.CudaTensor(#batch_images):copy(batch_images)
		batch_labels_gpu = torch.CudaTensor(#batch_labels):copy(batch_labels)
		if nn.gModule then
			batch_all_scores = model:forward({batch_images_gpu, batch_rois, scale0_rois})
		else
			batch_all_scores = model:forward({batch_images_gpu, batch_rois})
		end
		batch_scores=batch_all_scores[1]
		cost = HingeCriterion():setFactor(1 / numClasses):cuda():forward(batch_scores,batch_labels_gpu)

		table.insert(
			scale_scores, 
			(type(batch_scores) == 'table' and batch_scores[1] or batch_scores):float()
		)
		table.insert(scale_costs, cost)

		local batch_all_scores3 = makeContiguous(batch_all_scores[3]):clone()
		local batch_all_scores4 = makeContiguous(batch_all_scores[4]):clone()

		scale_outputs['output_prod_cls'] = scale_outputs['output_prod_cls'] or {}
		table.insert(
			scale_outputs['output_prod_cls'], 
			batch_all_scores[2]:view(1,-1,20):transpose(2, 3):float()
		)
		scale_outputs['output_prod_det'] = scale_outputs['output_prod_det'] or {}
		table.insert(
			scale_outputs['output_prod_det'], 
			batch_all_scores3:view(1,-1,20):transpose(2, 3):float()
		)
		scale_outputs['output_prod_det2'] = scale_outputs['output_prod_det2'] or {}
		table.insert(
			scale_outputs['output_prod_det2'], 
			batch_all_scores4:view(1,-1,20):transpose(2, 3):float()
		)
	end

	for output_field, output in pairs(scale_outputs) do
		outputs[output_field] = outputs[output_field] or {}
		table.insert(outputs[output_field], torch.cat(output, 1):mean(1):squeeze(1))
	end

	table.insert(scores, torch.cat(scale_scores, 1):mean(1))
	table.insert(labels, batch_labels:clone())
	table.insert(rois, scale0_rois:narrow(scale0_rois:dim(), 1, 4):clone()[1])
	
	collectgarbage()
	local output_string = string.format(
		"test  batch %04d  cost %.5f  speed %.2fs/img  TotalTime: %.1fmin", 
		batchIdx, 
		torch.FloatTensor(scale_costs):mean(), 
		torch.toc(tic_start)/batchIdx, 
		torch.toc(tic_start)/60
	)
	print(output_string)
end

local classLabels = {
	'aeroplane', 
	'bicycle', 
	'bird', 
	'boat', 
	'bottle', 
	'bus', 
	'car', 
	'cat', 
	'chair', 
	'cow', 
	'diningtable', 
	'dog', 
	'horse', 
	'motorbike', 
	'person', 
	'pottedplant', 
	'sheep', 
	'sofa', 
	'train', 
	'tvmonitor'
}

for output_field, output in pairs(outputs) do
	corloc_i = corloc(
		dataset[batch_loader.example_loader:getSubset(batch_loader.train)], 
		{output, rois}
	)
	corlocs[output_field]={}
	for i=1,20 do
		corlocs[output_field][classLabels[i]] = corloc_i[i]
	end
	corlocs_all[output_field]=corloc_i:mean()
end

local APtable = {}
local AP = dataset_tools.meanAP(torch.cat(scores, 1), torch.cat(labels, 1))
for i=1,20 do
	APtable[classLabels[i]] = AP[i]
end
table.insert(log, {
	training = false,
	mAP = AP:mean(),
	AP  = APtable,
	corlocs_all = corlocs_all,
	corlocs = corlocs,
})
print(log)

subset = batch_loader.example_loader:getSubset(batch_loader.train)
hdf5_save(
	opts.PATHS.SCORES_PATTERN:format(subset, SETTINGS.test_epoch_num),
	{
		subset = subset,
		meta = meta,
		rois = rois,
		labels = torch.cat(labels, 1),
		output = torch.cat(scores, 1),
		outputs = outputs,
	}
)

print('DONE:', torch.toc(tic_start), 'sec')
