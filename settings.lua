require 'torch'
require 'cudnn'
require 'salc'
require 'cusalc'
dofile 'layers/ContinuationSubset.lua'
dofile 'layers/ContinuationDetector.lua'
dofile 'layers/ReWeight.lua'
torch.setdefaulttensortype('torch.FloatTensor')

print(arg)

--setting gpu id
local device_id = arg[1]
cutorch.setDevice(device_id+1)

--setting for train and test
SETTINGS = {
	--common
	DATASET             = 'VOC2007',
	PROPOSALS           = arg[2],
	BASE_MODEL          = 'VGGF',
	model_path          = 'model/CMIL.lua',

	--training
	NUM_EPOCHS          = 20,
	LearningRate        = 5e-3,
	LearningRateAneal   = 5e-4,
	AnnealEpoch         = 10,
	DetRate             = 0.1,
	CO_TRAINING_PATTERN = 'Anneal0.63',
	annealFromEpoch3    = true,
	SUBSET              = 'trainval',
	CliqueNTOP          = 200,
	lambda              = 0.7,
	ifContinuation      = true,
	ContinuationFunc    = 'Log', --'Linear', 'Plinear', 'Sigmoid', 'Log', 'Exp' 

	--testing
	SUBSET_FOR_TESTING  = 'test',
	test_epoch_num      = 20
}
SETTINGS.RESULT_SAVE_FOLDER = SETTINGS.DATASET ..
	'/' .. SETTINGS.BASE_MODEL ..
	'/CMIL-' .. SETTINGS.PROPOSALS
print('SETTINGS:')
print(SETTINGS)
