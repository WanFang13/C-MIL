-- settings for path and models
dofile('settings.lua')
dofile('preprocess.lua')
dofile('util.lua')
dofile('dataset.lua')
dofile('layers/util.lua')
require 'optim'
dofile('fbnn_Optim.lua')

torch.manualSeed(opts.SEED)
cutorch.manualSeedAll(opts.SEED)
example_loader_options_preset = {
	training = {
		numRoisPerImage = 8192,
		subset = SETTINGS.SUBSET,
		hflips = true,
		numScales = 5,
	},
	evaluate = {
		numRoisPerImage = 8192,
		subset = SETTINGS.SUBSET,
		hflips = true,
		numScales = 1,
	}
}
opts.PATHS.MODEL = SETTINGS.model_path
if paths.extname(opts.PATHS.MODEL) == 'lua' then
	loaded = model_load(opts.PATHS.MODEL, opts)
	meta = {
		model_path = loaded.model_path,
		opts = opts,
		example_loader_options = example_loader_options_preset
	}
	log = {{meta = meta}}
else --resume
	opts.PATHS.MODEL = 'output/VOC2007/VGGF/CMIL-SSW/model_epoch16.h5'
	loaded = model_load(opts.PATHS.MODEL)
	meta = loaded.meta
	log = loaded.log
	previous_epoch = loaded.epoch
end
print('load model done.')

batch_loader = ParallelBatchLoader(
	ExampleLoader(
		dataset, 
		base_model.normalization_params, 
		opts.IMAGE_SCALES, 
		meta.example_loader_options
	)
):setBatchSize({training = 1, evaluate = 1})
print(meta)
print(model)

assert(model):cuda()
assert(criterion):cuda()
collectgarbage()
model:apply(function (x) x.for_each = x.apply end)
optimizer = nn.Optim(model, optimState)
optimalg = optim.sgd

log_details = {}
tic_start = torch.tic()
for epoch = (previous_epoch or 0) + 1, opts.NUM_EPOCHS do
	epoch_id = epoch
	if epoch > optimState_annealed.epoch then
		optimizer:setParameters(optimState_annealed)
	end
	batch_loader:training()
	model:training()
	batchIdx_global = nil
	tic = torch.tic()
	for batchIdx = 1, batch_loader:getNumBatches() -1 do
		if SETTINGS.ifContinuation then
			local nImg = batch_loader:getNumBatches()
			SETTINGS.lambda = getlambda(epoch-1+batchIdx/nImg, SETTINGS.NUM_EPOCHS, SETTINGS.ContinuationFunc)
		end
		
		batchIdx_global = batchIdx
		scale_batches = batch_loader:forward()[1]
		scale0_rois = scale_batches[1][2]:clone()
		batch_images, batch_rois, batch_labels = unpack(scale_batches[2])
		batch_images_gpu = torch.CudaTensor(#batch_images):copy(batch_images)
		batch_labels_gpu = torch.CudaTensor(#batch_labels):copy(batch_labels)
		batch_box_labels_gpu = torch.CudaTensor()
		cost = optimizer:optimize(
			optimalg, 
			{batch_images_gpu, batch_rois}, 
			{batch_labels_gpu, batch_box_labels_gpu}, 
			criterion
		)
		collectgarbage()
		local output_string = string.format(
			"epoch %02d  batch %04d  lambda %.2f  cost %.5f  speed %.2fs/img  TotalTime: %.1fmin",
			epoch, 
			batchIdx, 
			SETTINGS.lambda,
			cost, 
			torch.toc(tic)/batchIdx, 
			torch.toc(tic_start)/60
		)
		if batchIdx % 20 == 0 then
			print(output_string)
		end
	end

	if epoch % 4 == 0 or epoch == opts.NUM_EPOCHS or epoch == 1 then
		batch_loader:evaluate()
		model:evaluate()
		scores, labels, rois, costs, outputs, corlocs, corlocs_all = {},{},{},{},{},{},{}
		tic_val = torch.tic()
		for batchIdx = 1, batch_loader:getNumBatches() - 1 do
			
			scale_batches = batch_loader:forward()[1]
			scale0_rois = scale_batches[1][2]:clone()
			scale_outputs, scale_scores, scale_costs = {}, {}, {}
			for i = 2, #scale_batches do
				batch_images, batch_rois, batch_labels = unpack(scale_batches[i])
				batch_images_gpu = torch.CudaTensor(#batch_images):copy(batch_images)
				batch_labels_gpu = torch.CudaTensor(#batch_labels):copy(batch_labels)

				batch_all_scores = model:forward({batch_images_gpu, batch_rois})
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
				table.insert(outputs[output_field], torch.cat(output, 1):mean(1)[1])
			end

			table.insert(costs, torch.FloatTensor(scale_costs):mean())
			table.insert(scores, torch.cat(scale_scores, 1):mean(1))
			table.insert(labels, batch_labels:clone())
			table.insert(rois, scale0_rois:narrow(scale0_rois:dim(), 1, 4):clone()[1])
			local output_string = string.format(
				"val epoch %02d  batch %04d  cost %.5f  speed %.2fs/img  TotalTime: %.1fmin", 
				epoch, 
				batchIdx, 
				costs[#costs], 
				torch.toc(tic_val)/batchIdx, 
				torch.toc(tic_start)/60
			)
			if batchIdx % 20 == 0 then
				print(output_string)
			end
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
			epoch = epoch,
			mAP = AP:mean(),
			corlocs_all = corlocs_all,
			valCost = torch.FloatTensor(costs):mean(),
		})
		table.insert(log_details, {
			training = false,
			epoch = epoch,
			mAP = AP:mean(),
			AP  = APtable,
			corlocs = corlocs,
			corlocs_all = corlocs_all,
			valCost = torch.FloatTensor(costs):mean(),
		})
		print(log_details)
	end

	if epoch % 2 == 0 or epoch == opts.NUM_EPOCHS then
		model:clearState()
		model_save(opts.PATHS.CHECKPOINT_PATTERN:format(epoch), model, meta, epoch, log)
	end
	json_save(opts.PATHS.LOG, log)
	io.stderr:write('log in "', opts.PATHS.LOG, '"\n')
end
table.insert(log, log_details)
json_save(opts.PATHS.LOG, log)
io.stderr:write('details log in "', opts.PATHS.LOG, '"\n')