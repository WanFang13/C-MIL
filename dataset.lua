if opts.DATASET == 'VOC2007' or opts.DATASET == 'VOC2012' then
	dataset_tools = dofile('pascal_voc.lua')
	classLabels = dataset_tools.classLabels
	numClasses = dataset_tools.numClasses
end
--print(opts.PATHS.DATASET_CACHED)
print('Loading Dataset in:  ' .. opts.PATHS.DATASET_CACHED)
dataset = torch.load(opts.PATHS.DATASET_CACHED)
print('Dataset load done.')

dofile('parallel_batch_loader.lua')
dofile('example_loader.lua')

