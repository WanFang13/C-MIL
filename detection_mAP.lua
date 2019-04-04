-- settings for path and models
dofile('settings.lua')
dofile('preprocess.lua')

dofile('opts.lua')
dofile('util.lua')
dofile('dataset.lua')
threads = require 'threads'
print('Starting...')
local MATLAB = assert((#sys.execute('which matlab') > 0 and
		'matlab -nodisplay -r') or
		(#sys.execute('which octave') > 0 and 'octave --eval'),
		'matlab or octave not found in PATH')
local subset = 'test'


local i=1
local check_input = false
if arg[i+2] == '1' then
	opts.NMS_OVERLAP_THRESHOLD = 0.4
	opts.NMS_SCORE_THRESHOLD = 1e-4
	check_input = true
	output_field = opts.OUTPUT_FIELDS[1]
	--output_field = 'output_prod'
	prefix = '-cls'
	print('detect_mAP: ' .. opts.OUTPUT_FIELDS[1])
end
if arg[i+2] == '2' then
	opts.NMS_OVERLAP_THRESHOLD = 0.3
	opts.NMS_SCORE_THRESHOLD = 5e-3
	check_input = true
	output_field = opts.OUTPUT_FIELDS[2]
	prefix = '-det'
	print('detect_mAP: ' .. opts.OUTPUT_FIELDS[2])
end
if arg[i+2] == '3' then
	opts.NMS_OVERLAP_THRESHOLD = 0.3
	opts.NMS_SCORE_THRESHOLD = 5e-3
	check_input = true
	output_field = opts.OUTPUT_FIELDS[3]
	prefix = '-det2'
	print('detect_mAP: ' .. opts.OUTPUT_FIELDS[3])
end
if arg[i+2] == '12' then
	opts.NMS_OVERLAP_THRESHOLD = 0.3
	opts.NMS_SCORE_THRESHOLD = 5e-3
	check_input = true
	evalue_both = true
	output_field = 'output_clsdet'
	prefix = '-clsdet'
	print('detect_mAP: cls and det')
end

if not arg[i+2] or not check_input then
	print('Please choose the right output field (1 or 2)!')
	os.exit()
end



opts.SCORES_FILES = arg
--opts.SCORES_FILES = 1
rois = hdf5_load(opts.SCORES_FILES[i+1], 'rois')

scores = {}
if evalue_both then
	scores_1 = hdf5_load(opts.SCORES_FILES[i+1], 'outputs/' .. opts.OUTPUT_FIELDS[2])
	scores_2 = hdf5_load(opts.SCORES_FILES[i+1], 'outputs/' .. opts.OUTPUT_FIELDS[3])
	for exampleIdx = 1, #scores_1 do
		scores[exampleIdx] = scores_1[exampleIdx]:clone()
		scores[exampleIdx] = scores[exampleIdx]:add(scores_2[exampleIdx])/2
	end
else
	local i=2
	print('-------------------------')
	print(opts.SCORES_FILES[i])
	scores_i = hdf5_load(opts.SCORES_FILES[i], 'outputs/' .. output_field)
	for exampleIdx = 1, #scores_i do
		scores[exampleIdx] = (scores[exampleIdx] or scores_i[exampleIdx]:clone():zero()):add(scores_i[exampleIdx]:div(#opts.SCORES_FILES))
	end
end

local detrespath = dataset_tools.package_submission(
	opts.PATHS.VOC_DEVKIT_VOCYEAR, 
	dataset,
	opts.DATASET, 
	subset, 
	'comp4_det', 
	rois, 
	scores, 
	nms_mask(rois, scores, opts.NMS_OVERLAP_THRESHOLD, opts.NMS_SCORE_THRESHOLD)
)
local opts = opts

if dataset[subset].objectBoxes == nil then
	print('detection mAP cannot be computed for ' .. opts.DATASET .. '. Quitting.')
	print(('VOC submission saved in "%s/results-%s-%s-%s.tar.gz"'):format(opts.PATHS.DATA, opts.DATASET, 'comp4_det', subset))
	os.exit(0)
end

res = {[output_field] = {_mean = nil, by_class = {}}}
APs = torch.FloatTensor(numClasses):zero()

local imgsetpath = paths.tmpname()
--print('sed \'s/$/ -1/\' %s > %s'):format(paths.concat(opts.PATHS.VOC_DEVKIT_VOCYEAR, 'ImageSets', 'Main', subset .. '.txt'), imgsetpath)
os.execute(('sed \'s/$/ -1/\' %s > %s'):format(paths.concat(opts.PATHS.VOC_DEVKIT_VOCYEAR, 'ImageSets', 'Main', subset .. '.txt'), imgsetpath)) -- hack for octave

jobQueue = threads.Threads(numClasses)
for classLabelInd, classLabel in ipairs(classLabels) do
	jobQueue:addjob(function()
		--print('%s "oldpwd = pwd; cd(\'%s\'); addpath(fullfile(pwd, \'VOCcode\')); VOCinit; cd(oldpwd); VOCopts.testset = \'%s\'; VOCopts.detrespath = \'%s\'; VOCopts.imgsetpath = \'%s\'; classLabel = \'%s\'; [rec, prec, ap] = VOCevaldet(VOCopts, \'comp4\', classLabel, false); dlmwrite(sprintf(VOCopts.detrespath, \'resu4\', classLabel), ap); quit;"'):format(MATLAB, paths.dirname(opts.PATHS.VOC_DEVKIT_VOCYEAR), subset, detrespath, imgsetpath, classLabel)
		os.execute(('%s "oldpwd = pwd; cd(\'%s\'); addpath(fullfile(pwd, \'VOCcode\')); VOCinit; cd(oldpwd); VOCopts.testset = \'%s\'; VOCopts.detrespath = \'%s\'; VOCopts.imgsetpath = \'%s\'; classLabel = \'%s\'; [rec, prec, ap] = VOCevaldet(VOCopts, \'comp4\', classLabel, false); dlmwrite(sprintf(VOCopts.detrespath, \'resu4\', classLabel), ap); quit;"'):format(MATLAB, paths.dirname(opts.PATHS.VOC_DEVKIT_VOCYEAR), subset, detrespath, imgsetpath, classLabel))
		return tonumber(io.open(detrespath:format('resu4', classLabel)):read('*all'))
	end, function(ap) res[output_field].by_class[classLabel] = ap; APs[classLabelInd] = ap; end)
end
jobQueue:synchronize()
os.execute('[ -t 1 ] && reset')

res[output_field]._mean = APs:mean()

json_save(opts.PATHS.DETECTION_MAP:format(SETTINGS.test_epoch_num, prefix), res)
print('result in ' .. opts.PATHS.DETECTION_MAP:format(SETTINGS.test_epoch_num, prefix))
