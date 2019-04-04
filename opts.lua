local DATASETS_PATH   = 'data/datasets'
local DATA_CNN_MODELS = 'data/models'
local RESULT_SAVE     = 'output/' .. SETTINGS.RESULT_SAVE_FOLDER

local temp = io.open(RESULT_SAVE)
if not temp then
  os.execute('mkdir ' .. RESULT_SAVE)
end

PATHS = {
	EXTERNAL = {
		PRETRAINED_MODEL_VGGF =   {
			PROTOTXT   = paths.concat(DATA_CNN_MODELS, 'VGG_CNN_F_deploy.prototxt'),
			CAFFEMODEL = paths.concat(DATA_CNN_MODELS, 'VGG_CNN_F.caffemodel')
		},

		PRETRAINED_MODEL_VGG16 =   {
			PROTOTXT   = paths.concat('../py-faster-rcnn/data/imagenet_models/VGG16_deploy.prototxt'),
			CAFFEMODEL = paths.concat('../py-faster-rcnn/data/imagenet_models/VGG16.v2.caffemodel')
		},

		PRETRAINED_MODEL_AlexNet =   {
			PROTOTXT   = paths.concat('../py-faster-rcnn/data/imagenet_models/AlexNet_deploy.prototxt'),
			CAFFEMODEL = paths.concat('../py-faster-rcnn/data/imagenet_models/AlexNet.v2.caffemodel')
		},

		SSW_VOC2007 =  {
			trainval = paths.concat(DATASETS_PATH, 'SelectiveSearchVOC2007trainval.mat'),
			test     = paths.concat(DATASETS_PATH, 'SelectiveSearchVOC2007test.mat')
		},

		EB_VOC2007 =  {
			trainval = paths.concat(DATASETS_PATH, 'EdgeBoxesVOC2007trainval.mat'),
			test     = paths.concat(DATASETS_PATH, 'EdgeBoxesVOC2007test.mat')
		},

		SSW_VOC2012 =  {
			train     = paths.concat(DATASETS_PATH, 'voc_2012_train.mat'),
			val       = paths.concat(DATASETS_PATH, 'voc_2012_val.mat'),
			trainval  = paths.concat(DATASETS_PATH, 'voc_2012_trainval.mat'),
			test      = paths.concat(DATASETS_PATH, 'voc_2012_test.mat')
		},
		
		VOC_DEVKIT_VOCYEAR =  {
			VOC2007 = paths.concat(DATASETS_PATH, 'VOCdevkit_2007/VOC2007'),
			VOC2010 = paths.concat(DATASETS_PATH, 'VOCdevkit_2010/VOC2010'),
			VOC2012 = paths.concat(DATASETS_PATH, 'VOCdevkit_2012/VOC2012')
		}
	},
	
	BASE_MODEL_CACHED = {
		VGGF      = paths.concat(DATA_CNN_MODELS, 'VGG_CNN_F.t7'),
		GoogleNet = paths.concat(DATA_CNN_MODELS, 'GoogleNet.t7'),
		VGG16     = paths.concat(DATA_CNN_MODELS, 'VGG16.t7'),
		AlexNet    = paths.concat(DATA_CNN_MODELS, 'AlexNet.t7')
	},
	
	BASE_MODEL_CACHED_ZB = {
		VGGF      = paths.concat(DATA_CNN_MODELS, 'VGG_CNN_F_zb.t7'),
		GoogleNet = paths.concat(DATA_CNN_MODELS, 'GoogleNet_zb.t7')
	},

	DATASET_CACHED_PATTERN         = paths.concat(DATASETS_PATH, '%s_%s.t7'),
	CHECKPOINT_PATTERN             = paths.concat(RESULT_SAVE, 'model_epoch%02d.h5'),
  CHECKPOINT_PATTERN_ROOT_FOLDER = paths.concat(RESULT_SAVE, ''),
	LOG                            = paths.concat(RESULT_SAVE, 'log.json'),
	SCORES_PATTERN                 = paths.concat(RESULT_SAVE, 'scores_%s_epoch%02d.h5'),
	CORLOC                         = paths.concat(RESULT_SAVE, 'corloc.json'),
	DETECTION_MAP                  = paths.concat(RESULT_SAVE, 'detection_mAP_epoch%02d%s.json')
}

local BASE_MODEL_CACHED_  = PATHS.BASE_MODEL_CACHED[SETTINGS.BASE_MODEL]
local BASE_MODEL_CACHEDZB = PATHS.BASE_MODEL_CACHED_ZB[SETTINGS.BASE_MODEL]

opts = {
	ROI_FACTOR            = 1.8,
	SEED                  = 666,
	NMS_OVERLAP_THRESHOLD = 0.3,
	NMS_SCORE_THRESHOLD   = 5e-3,
	--IMAGE_SCALES = {{608, 800}, {368, 480}, {432, 576}, {528, 688}, {656, 864}, {912, 1200}}
	IMAGE_SCALES          = {{608, 800}, {496, 656}, {400, 544}, {720, 960}, {864, 1152}}, 
	NUM_SCALES            = 5,
	OUTPUT_FIELDS         = {'output_prod_cls', 'output_prod_det', 'output_prod_det2'},
	NUM_EPOCHS            = SETTINGS.NUM_EPOCHS,
	DATASET               = SETTINGS.DATASET,
	BASE_MODEL            = SETTINGS.BASE_MODEL,
	SUBSET                = SETTINGS.SUBSET,
	
	PATHS = {
		--MODEL = arg[1],
		MODEL               = nil,
		CHECKPOINT_PATTERN  = PATHS.CHECKPOINT_PATTERN,
		LOG                 = PATHS.LOG,
		SCORES_PATTERN      = PATHS.SCORES_PATTERN,
		PROPOSALS           = PATHS.EXTERNAL[SETTINGS.PROPOSALS .. '_' .. SETTINGS.DATASET],
		CORLOC              = PATHS.CORLOC,
		DETECTION_MAP       = PATHS.DETECTION_MAP,
		RUN_STATS_PATTERN   = PATHS.RUN_STATS_PATTERN,
		BASE_MODEL_RAW      = PATHS.EXTERNAL['PRETRAINED_MODEL_' .. SETTINGS.BASE_MODEL],
		VOC_DEVKIT_VOCYEAR  = PATHS.EXTERNAL.VOC_DEVKIT_VOCYEAR[SETTINGS.DATASET],
		DATASET_CACHED      = PATHS.DATASET_CACHED_PATTERN:format(SETTINGS.DATASET,SETTINGS.PROPOSALS),
		BASE_MODEL_CACHED   = SETTINGS.ZB_DEBUG and BASE_MODEL_CACHEDZB or BASE_MODEL_CACHED_
	}
}
