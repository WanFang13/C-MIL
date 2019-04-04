#include <stdlib.h>
#include <assert.h>
struct node
{
	float data;
	int ind;
};

int struct_cmp_descend(const void *a, const void *b) 
{
	return (*(node *)a).data < (*(node *)b).data ? 1 : -1;
}

int struct_cmp_ascend(const void *a, const void *b) 
{
	return (*(node *)a).data > (*(node *)b).data ? 1 : -1;
}


void node_qsort_descend(float *data, int *sort_inds, int nData)
{
	struct node node_data[nData];
	for(int i = 0; i < nData;++i)
	{
		node_data[i].data = data[i];
		node_data[i].ind  = i;
	}
	qsort(node_data, nData, sizeof(node_data[0]), struct_cmp_descend);
	for(int i = 0; i < nData;++i)
	{
		data[i]      = node_data[i].data;
		sort_inds[i] = node_data[i].ind;
	}
}

void node_qsort_ascend(float *data, int *sort_inds, int nData)
{
	struct node node_data[nData];
	for(int i = 0; i < nData;++i)
	{
		node_data[i].data = data[i];
		node_data[i].ind  = i;
	}
	qsort(node_data, nData, sizeof(node_data[0]), struct_cmp_ascend);
	for(int i = 0; i < nData;++i)
	{
		data[i]      = node_data[i].data;
		sort_inds[i] = node_data[i].ind;
	}
}

//Begin: split_function //==================================================================
template <typename Dtype>
__global__ void split_input(Dtype *Cls, Dtype *ClsW, Dtype *output, Dtype *outputW,
														Dtype *inds, const int inds_count)
{
  CUDA_KERNEL_LOOP(index, inds_count)
  {
		int n = index;
		// check whether is the beginning of the local rois or not
		if( n==0 || inds[(n-1)*3+1]!=inds[n*3+1]  )
		{
			// if is beginning then do 
			float temp_count=0;
			for(int i=n;i<inds_count;i++)
			{
				int out_id = inds[i*3+1] -1;
				int in_id  = inds[i*3+2] -1;
				// accumulate the values of local_inds i
				output[out_id]  += Cls[in_id];
				outputW[out_id] += ClsW[in_id];
				temp_count +=1;
				// check local consistency
				if(i+1==inds_count || inds[(i+1)*3+1]!=inds[i*3+1]) {break;}
			}
			// average pooling
			outputW[ (int)(inds[n*3+1]-1)] /= temp_count;
			output[ (int)(inds[n*3+1]-1)] /= temp_count;
		}
  }
}
static int cusalc_ME_SplitInput(lua_State *L)
{
	// input L: slpit input datas into output datas
	// THCState *state, THCudaTensor *input, THCudaTensor *output,
	// THCudaTensor *inds, int inds_count, int dim
	
	THCState *state       = getCutorchState(L);
	THCudaTensor *Cls     = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *ClsW    = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *output  = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	THCudaTensor *outputW = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
	THCudaTensor *inds    = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
	long inds_count       = lua_tonumber(L, 7);

	float* fCls     = THCudaTensor_data(state, Cls);
	float* fClsW    = THCudaTensor_data(state, ClsW);
	float* foutput  = THCudaTensor_data(state, output);
	float* foutputW = THCudaTensor_data(state, outputW);
	float* finds    = THCudaTensor_data(state, inds);
  split_input<<<GET_BLOCKS(inds_count), CUDA_NUM_THREADS>>>(fCls, fClsW, foutput, foutputW, finds, inds_count);
	
	return 1;
}
//End: split_function //==================================================================


//Begin: merge_input_function_fast //==================================================================
template <typename Dtype>
__global__ void merge_input_fast( const Dtype *Cls, const Dtype *ClsW, 
																	Dtype *output, Dtype *outputW, Dtype *inds, 
																	const int dim, const int inds_count)
{
  CUDA_KERNEL_LOOP(index, inds_count)
  {
		int n = index;
		// check whether is the beginning of the local rois or not
		if( n==0 || inds[(n-1)*3+1]!=inds[n*3+1]  )
		{
			// if is beginning then do 
			float temp_count=0;
			int out_id_old = inds[n*3+1]-1;
			for(int i=n;i<inds_count;i++)
			{
				int out_id = inds[i*3+1] -1;
				int in_id  = inds[i*3+2] -1;
				// accumulate the values of local_inds i
				for(int j=0;j<dim;j++)
				{
					output[ out_id*dim+j] += Cls[ in_id*dim+j];
					outputW[out_id*dim+j] += ClsW[in_id*dim+j];
				}
				assert(out_id_old == out_id);
				out_id_old = out_id;
				temp_count +=1;
				// check local consistency
				if(i+1==inds_count || inds[(i+1)*3+1]!=inds[i*3+1]) {break;}
			}
			// average pooling
			for(int k=0;k<dim;k++)
			{
				outputW[out_id_old*dim+k] /= temp_count;
				output[ out_id_old*dim+k] /= temp_count;
			}
		}
  }
}

static int cusalc_ME_MergeInputFast(lua_State *L)
{
	// input L: slpit input datas into output datas
	// THCState *state, THCudaTensor *input, THCudaTensor *output,
	// THCudaTensor *inds, int inds_count, int dim
	
	THCState *state       = getCutorchState(L);
	THCudaTensor *Cls     = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *ClsW    = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *output  = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	THCudaTensor *outputW = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
	THCudaTensor *inds    = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
	long dim              = lua_tonumber(L, 7);
	long inds_count       = lua_tonumber(L, 8);
	
	float* fCls     = THCudaTensor_data(state, Cls);
	float* fClsW    = THCudaTensor_data(state, ClsW);
	float* foutput  = THCudaTensor_data(state, output);
	float* foutputW = THCudaTensor_data(state, outputW);
	float* finds    = THCudaTensor_data(state, inds);
  merge_input_fast<<<GET_BLOCKS(inds_count), CUDA_NUM_THREADS>>>(fCls, fClsW, foutput, foutputW, finds, dim, inds_count);
	
	return 1;
}
//End: merge_input_function_fast //==================================================================








//Begin: merge_grad_function //==================================================================
template <typename Dtype>
__global__ void merge_grads(const Dtype *inputCls, const Dtype *inputClsW, 
														Dtype *gradCls, Dtype *gradClsW, const Dtype *inds, 
														const int inds_count, const Dtype *local_nums)
{
  CUDA_KERNEL_LOOP(i, inds_count)
  {
		int n      = i;
		int in_id  = inds[n*3+1]-1;  // splitted grads inds
		int out_id = inds[n*3+2]-1;	 // merge grads inds
		
		gradCls[out_id]  = inputCls[in_id]/local_nums[in_id];
		gradClsW[out_id] = inputClsW[in_id]/local_nums[in_id];
		//printf("in_id=%d, local_nums[in_id]=%.2f \n", in_id, local_nums[in_id]);
  }
}
template <typename Dtype>
__global__ void comput_local_nums(Dtype *inds, Dtype *local_nums, const int inds_count)
{
  CUDA_KERNEL_LOOP(index, inds_count)
  {
		int n = index;
		// check whether is the beginning of the local rois or not
		if( n==0 || inds[(n-1)*3+1]!=inds[n*3+1]  )
		{
			// if is beginning then do 
			local_nums[(int)inds[n*3+1]-1]=0;
			for(int i=n;i<inds_count;i++)
			{
				local_nums[(int)inds[n*3+1]-1] +=1;
				// check local consistency
				if(i+1==inds_count || inds[(i+1)*3+1]!=inds[i*3+1]) {break;}
			}
			//printf("inds[n*3+1]-1=%d, local_nums[inds[n*3+1]-1]=%.2f \n", inds[n*3+1]-1, local_nums[(int)inds[n*3+1]-1]);
		}
  }
}
static int cusalc_ME_MergeGrads(lua_State *L)
{
	// input L: merge input grads into output grads
	// THCState *state, THCudaTensor *input, THCudaTensor *output,
	// THCudaTensor *inds, int inds_count, int dim
	
	THCState *state = getCutorchState(L);
	THCudaTensor *gradCls   = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *gradClsW  = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *inputCls  = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	THCudaTensor *inputClsW = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
	THCudaTensor *inds      = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
	long inds_count         = lua_tonumber(L, 7);
	long nLocal             = lua_tonumber(L, 8);

	float* fgradCls   = THCudaTensor_data(state, gradCls);
	float* fgradClsW  = THCudaTensor_data(state, gradClsW);
	float* finputCls  = THCudaTensor_data(state, inputCls);
	float* finputClsW = THCudaTensor_data(state, inputClsW);
	float* finds      = THCudaTensor_data(state, inds);
	float* local_nums;// = (float*)malloc(nLocal*sizeof(float));
	
	cudaMalloc((void**)&local_nums, nLocal*sizeof(float));
	
  comput_local_nums<<<GET_BLOCKS(inds_count), CUDA_NUM_THREADS>>>(finds, local_nums, inds_count);
  merge_grads<<<GET_BLOCKS(inds_count), CUDA_NUM_THREADS>>>(finputCls, finputClsW, fgradCls, fgradClsW,
																														finds, inds_count, local_nums);
	
	cudaFree(local_nums);
	return 1;
}
//End: merge_grad_function //==================================================================	 






//Begin: merge_grad_fast_function //==================================================================
template <typename Dtype>
__global__ void merge_grads_fast(const Dtype *inputCls, const Dtype *inputClsW, 
														Dtype *gradCls, Dtype *gradClsW, const Dtype *inds, 
														const int inds_count, const Dtype *local_nums, int dim)
{
  CUDA_KERNEL_LOOP(i, inds_count)
  {
		int n      = i;
		int in_id  = inds[n*3+1]-1;  // splitted grads inds
		int out_id = inds[n*3+2]-1;	 // merge grads inds
		
		for(int i=0;i<dim;i++)
		{
			gradCls[out_id*dim+i]  = inputCls[in_id*dim+i]/local_nums[in_id];
			gradClsW[out_id*dim+i] = inputClsW[in_id*dim+i]/local_nums[in_id];
		}
		//printf("in_id=%d, local_nums[in_id]=%.2f \n", in_id, local_nums[in_id]);
  }
}
static int cusalc_ME_MergeGradsFast(lua_State *L)
{
	// input L: merge input grads into output grads
	// THCState *state, THCudaTensor *input, THCudaTensor *output,
	// THCudaTensor *inds, int inds_count, int dim
	
	THCState *state = getCutorchState(L);
	THCudaTensor *gradCls   = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *gradClsW  = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *inputCls  = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	THCudaTensor *inputClsW = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
	THCudaTensor *inds      = (THCudaTensor*)luaT_checkudata(L, 6, "torch.CudaTensor");
	long inds_count         = lua_tonumber(L, 7);
	long nLocal             = lua_tonumber(L, 8);
	long dim                = lua_tonumber(L, 9);

	float* fgradCls   = THCudaTensor_data(state, gradCls);
	float* fgradClsW  = THCudaTensor_data(state, gradClsW);
	float* finputCls  = THCudaTensor_data(state, inputCls);
	float* finputClsW = THCudaTensor_data(state, inputClsW);
	float* finds      = THCudaTensor_data(state, inds);
	float* local_nums;// = (float*)malloc(nLocal*sizeof(float));
	
	cudaMalloc((void**)&local_nums, nLocal*sizeof(float));
	
  comput_local_nums<<<GET_BLOCKS(inds_count), CUDA_NUM_THREADS>>>(finds, local_nums, inds_count);
  merge_grads_fast<<<GET_BLOCKS(inds_count), CUDA_NUM_THREADS>>>(finputCls, finputClsW, fgradCls, fgradClsW,
																														finds, inds_count, local_nums, dim);
	
	cudaFree(local_nums);
	return 1;
}
//End: merge_grad_fast_function //==================================================================	 










//Begin: merge_boxes_score_function //==================================================================
template <typename Dtype>
__global__ void merge_scores(Dtype *input, Dtype *output, Dtype *inds,
														const int inds_count, const Dtype *local_nums)
{
  CUDA_KERNEL_LOOP(i, inds_count)
  {
		int n      = i;
		int in_id  = inds[n*3+1]-1;  // splitted grads inds
		int out_id = inds[n*3+2]-1;	 // merge grads inds

		output[out_id] = input[in_id];
			 //if(local_nums[in_id]==0){printf("local_nums == 0!!\n");}
		//printf("in_id=%d, local_nums[in_id]=%.2f \n", in_id, local_nums[in_id]);
  }
}
static int cusalc_ME_MergeScores(lua_State *L)
{
	// input L: merge input splitted boxes scores 
	// THCState *state, THCudaTensor *input, THCudaTensor *output,
	// THCudaTensor *inds, int inds_count, int dim
	
	THCState *state = getCutorchState(L);
	THCudaTensor *input   = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *output  = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *inds    = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	long inds_count       = lua_tonumber(L, 5);
	long nLocal           = lua_tonumber(L, 6);

	float* finput  = THCudaTensor_data(state, input);
	float* foutput = THCudaTensor_data(state, output);
	float* finds   = THCudaTensor_data(state, inds);
	float* local_nums;// = (float*)malloc(nLocal*sizeof(float));
	
	cudaMalloc((void**)&local_nums, nLocal*sizeof(float));
	
  comput_local_nums<<<GET_BLOCKS(inds_count), CUDA_NUM_THREADS>>>(finds, local_nums, inds_count);
  merge_scores<<<GET_BLOCKS(inds_count), CUDA_NUM_THREADS>>>(finput, foutput, finds, inds_count, local_nums);
	
	cudaFree(local_nums);
	return 1;
}
template <typename Dtype>
__global__ void merge_scores_vis(Dtype *input, Dtype *output, Dtype *cls_id, Dtype *inds,
														const int inds_count, const Dtype *local_nums, int ci)
{
  CUDA_KERNEL_LOOP(i, inds_count)
  {
		int n      = i;
		int in_id  = inds[n*3+1]-1;  // splitted grads inds
		int out_id = inds[n*3+2]-1;	 // merge grads inds

		output[out_id] = input[in_id];
			 //if(local_nums[in_id]==0){printf("local_nums == 0!!\n");}
		//printf("in_id=%d, local_nums[in_id]=%.2f \n", in_id, local_nums[in_id]);
		assert(cls_id[out_id] == 0);
		cls_id[out_id] = ci;
  }
}
static int cusalc_ME_MergeScoresVis(lua_State *L)
{
	// input L: merge input splitted boxes scores 
	// THCState *state, THCudaTensor *input, THCudaTensor *output,
	// THCudaTensor *inds, int inds_count, int dim
	
	THCState *state = getCutorchState(L);
	THCudaTensor *input   = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *output  = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *cls_id  = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	THCudaTensor *inds    = (THCudaTensor*)luaT_checkudata(L, 5, "torch.CudaTensor");
	long inds_count       = lua_tonumber(L, 6);
	long nLocal           = lua_tonumber(L, 7);
	long ci               = lua_tonumber(L, 8);

	float* finput  = THCudaTensor_data(state, input);
	float* foutput = THCudaTensor_data(state, output);
	float* fcls_id = THCudaTensor_data(state, cls_id);
	float* finds   = THCudaTensor_data(state, inds);
	float* local_nums;// = (float*)malloc(nLocal*sizeof(float));
	
	cudaMalloc((void**)&local_nums, nLocal*sizeof(float));
	
  comput_local_nums<<<GET_BLOCKS(inds_count), CUDA_NUM_THREADS>>>(finds, local_nums, inds_count);
  merge_scores_vis<<<GET_BLOCKS(inds_count), CUDA_NUM_THREADS>>>(finput, foutput, fcls_id, finds, inds_count, local_nums,ci);
	
	cudaFree(local_nums);
	return 1;
}
//End: merge_boxes_score_function //==================================================================	 








//Begin: merge_boxes_score_fast_function //==================================================================
template <typename Dtype>
__global__ void merge_scores_fast(Dtype *input, Dtype *output, Dtype *inds,
														const int inds_count, const Dtype *local_nums, int dim)
{
  CUDA_KERNEL_LOOP(i, inds_count)
  {
		int n      = i;
		int in_id  = inds[n*3+1]-1;  // splitted grads inds
		int out_id = inds[n*3+2]-1;	 // merge grads inds
		
		for(int i=0;i<dim;i++)
		{
			output[out_id*dim+i] = input[in_id*dim+i];
		}
			 //if(local_nums[in_id]==0){printf("local_nums == 0!!\n");}
		//printf("in_id=%d, local_nums[in_id]=%.2f \n", in_id, local_nums[in_id]);
  }
}
static int cusalc_ME_MergeScoresFast(lua_State *L)
{
	// input L: merge input splitted boxes scores 
	// THCState *state, THCudaTensor *input, THCudaTensor *output,
	// THCudaTensor *inds, int inds_count, int dim
	
	THCState *state = getCutorchState(L);
	THCudaTensor *input  = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *output = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *inds   = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	long inds_count      = lua_tonumber(L, 5);
	long nLocal          = lua_tonumber(L, 6);
	long dim             = lua_tonumber(L, 7);

	float* finput  = THCudaTensor_data(state, input);
	float* foutput = THCudaTensor_data(state, output);
	float* finds   = THCudaTensor_data(state, inds);
	float* local_nums;// = (float*)malloc(nLocal*sizeof(float));
	
	cudaMalloc((void**)&local_nums, nLocal*sizeof(float));
	
  comput_local_nums<<<GET_BLOCKS(inds_count), CUDA_NUM_THREADS>>>(finds, local_nums, inds_count);
  merge_scores_fast<<<GET_BLOCKS(inds_count), CUDA_NUM_THREADS>>>(finput, foutput, finds, inds_count, local_nums, dim);
	
	cudaFree(local_nums);
	return 1;
}
//End: merge_boxes_score_fast_function //==================================================================	 









//Begin: Local Consistency functions //==================================================================
// Compute Distance
template <typename Dtype>
__global__ void compute_distance(const Dtype *rois, Dtype *dist, 
																 const int roi_num, const int roi_dim,
																 const float x_max, const float y_max,
																 const int dist_type)
{
  CUDA_KERNEL_LOOP(index, roi_num*roi_num)
  {
		int n = index / roi_num;
		int c = index % roi_num;
		if(dist_type==1)
		{
			//compute overlap
			float a1 = (rois[n*roi_dim+2]-rois[n*roi_dim+0] +1) * (rois[n*roi_dim+3]-rois[n*roi_dim+1] +1);
			float a2 = (rois[c*roi_dim+2]-rois[c*roi_dim+0] +1) * (rois[c*roi_dim+3]-rois[c*roi_dim+1] +1);
			
			float xx1 = rois[n*roi_dim+0]>rois[c*roi_dim+0] ? rois[n*roi_dim+0] : rois[c*roi_dim+0];
			float yy1 = rois[n*roi_dim+1]>rois[c*roi_dim+1] ? rois[n*roi_dim+1] : rois[c*roi_dim+1];
			float xx2 = rois[n*roi_dim+2]<rois[c*roi_dim+2] ? rois[n*roi_dim+2] : rois[c*roi_dim+2];
			float yy2 = rois[n*roi_dim+3]<rois[c*roi_dim+3] ? rois[n*roi_dim+3] : rois[c*roi_dim+3];
			
			float w = xx2-xx1+1>0 ? xx2-xx1+1 : 0;
			float h = yy2-yy1+1>0 ? yy2-yy1+1 : 0;
			float inter = w*h;
		
			dist[n*roi_num +c] = inter / (a1 + a2 - inter);
		}
		else if(dist_type == 2)
		{
			float x1 = (rois[n*roi_dim+2]+rois[n*roi_dim+0])/2.0/x_max;
			float y1 = (rois[n*roi_dim+3]+rois[n*roi_dim+1])/2.0/y_max;
			float x2 = (rois[c*roi_dim+2]+rois[c*roi_dim+0])/2.0/x_max;
			float y2 = (rois[c*roi_dim+3]+rois[c*roi_dim+1])/2.0/y_max;
			dist[n*roi_num +c] = sqrt(pow((x1-x2),2)+pow((y1-y2),2)); // Euclidean Distance
			/*printf("dist[%d,%d] = %f\n", n, c, dist[n*num +c]);*/
		}
		else if(dist_type == 3)
		{
			//compute overlap
			float a1 = (rois[n*roi_dim+2]-rois[n*roi_dim+0] +1) * (rois[n*roi_dim+3]-rois[n*roi_dim+1] +1);
			float a2 = (rois[c*roi_dim+2]-rois[c*roi_dim+0] +1) * (rois[c*roi_dim+3]-rois[c*roi_dim+1] +1);
			
			float xx1 = rois[n*roi_dim+0]>rois[c*roi_dim+0] ? rois[n*roi_dim+0] : rois[c*roi_dim+0];
			float yy1 = rois[n*roi_dim+1]>rois[c*roi_dim+1] ? rois[n*roi_dim+1] : rois[c*roi_dim+1];
			float xx2 = rois[n*roi_dim+2]<rois[c*roi_dim+2] ? rois[n*roi_dim+2] : rois[c*roi_dim+2];
			float yy2 = rois[n*roi_dim+3]<rois[c*roi_dim+3] ? rois[n*roi_dim+3] : rois[c*roi_dim+3];
			
			float w = xx2-xx1+1>0 ? xx2-xx1+1 : 0;
			float h = yy2-yy1+1>0 ? yy2-yy1+1 : 0;
			float inter = w*h;
		
			float temp = inter / (a1 + a2 - inter);
			if(temp<0.05)
			{
				dist[n*roi_num +c] = 21;
			}
			else
			{
				dist[n*roi_num +c] = 1/temp - 1;
			}
			
		}
		else
		{
			printf("error: Unknow Distence!\n");
		}
  }
}
// Compute Distance
static int cusalc_ME_ComputeDistance(lua_State *L)
{
	// input L: Compute distance matrix by input rois
	// THCState *state, THCudaTensor *rois, THCudaTensor *dist,
	// int roi_num, int roi_dim, x_max, y_max, dist_type
	
	THCState *state = getCutorchState(L);
	THCudaTensor *rois = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *dist = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	long roi_num       = lua_tonumber(L, 4);
	long roi_dim       = lua_tonumber(L, 5);
	float x_max        = lua_tonumber(L, 6);
	float y_max        = lua_tonumber(L, 7);
	long dist_type     = lua_tonumber(L, 8);  // 1:consistency    2: clustering

	float* frois = THCudaTensor_data(state, rois);
	float* fdist = THCudaTensor_data(state, dist);
	
  compute_distance<<<GET_BLOCKS(roi_num), CUDA_NUM_THREADS>>>(frois, fdist, roi_num, roi_dim, 
																													x_max, y_max, dist_type);
	
	return 1;
}
//End: Local Consistency functions //==================================================================	











//Begin: Local Clustering  //==================================================================
template <typename Dtype>
__global__ void compute_rho(const Dtype *dist, Dtype *rho, 
														const int num, const Dtype dc)
{
  CUDA_KERNEL_LOOP(index, num)
  {
		int n = index;
		rho[n] = 0;
		for(int i=0;i<num;i++)
		{
			rho[n] += exp(-pow(dist[n*num+i]/dc, 2));
		}
		rho[n] -=1;
		//printf("d_rho[%d]=%.5f\n", n, rho[n]);
  }
}
template <typename Dtype>
__global__ void compute_delta(const Dtype *dist, const int *ord_rho, 
														  Dtype *delta,  int *nneigh, 
															const int num, const float max_dist)
{
  CUDA_KERNEL_LOOP(index, num)
  {
		int n = index;
		int flag=0;
		
		delta[ord_rho[n]] = max_dist;
		for(int i=0; i<n; i++)
		{
			float temp_dist = dist[ord_rho[n]*num+ord_rho[i]];
			//if(n<10)
			//printf("[%d]: --- temp_dist = %.5f\n", ord_rho[n], temp_dist);
			if(temp_dist < delta[ord_rho[n]])
			{
				delta[ ord_rho[n]] = temp_dist;
				nneigh[ord_rho[n]] = ord_rho[i];
				flag += 1;
			}
		}
		if(!flag)
		{
			nneigh[ord_rho[n]] = -1;
		}
		//printf("n=%d\n", n);
		//printf("delta[%d] = %.10f,   nneigh[%d] = %d\n", ord_rho[n], delta[ ord_rho[n]], ord_rho[n], nneigh[ord_rho[n]]);
	}
}
template <typename Dtype>
__global__ void compute_gamma(const Dtype *rho, const Dtype *delta, 
														  Dtype *gamma, const int num)
{
  CUDA_KERNEL_LOOP(index, num)
  {
		int n = index;

		gamma[n] = rho[n] * delta[n];
		//printf("rho[%d]=%.4f,   delta[%d]=%.4f,   gamma[%d]=%.4f\n", n, rho[n], n, delta[n], n, gamma[n]);
	}
}
void compute_icl(const int *ord_gamma, const float *gamma,
								 int *icl, int *nClass, 
								 const int max_nCluster, const int flag)
{
	icl[0] = ord_gamma[0];
	*nClass = 1;
	if(flag)
	{
		for(int i=1;i<max_nCluster;i++)
		{
			if(gamma[i]/gamma[0] > 0.2  && gamma[i]/gamma[i+1] >2)
			{
				icl[*nClass] = ord_gamma[i];
				*nClass += 1;
			}
		}
	}
}
void compute_icl_vis(const int *ord_gamma, const float *gamma,
								 const int *ord_delta, int *icl, int *nClass, 
								 const int max_nCluster, const int flag)
{
	icl[0] = ord_gamma[0];
	*nClass = 1;
	printf("gamma[0] = %.3f\n", gamma[0]);
	if(flag)
	{
		for(int i=1;i<max_nCluster;i++)
		{
			if(gamma[i]/gamma[0] > 0  && gamma[i]/gamma[i+1] >1.2)
			{
				icl[*nClass] = ord_gamma[i];
				*nClass += 1;
				//printf("gamma[%d] = %.3f\n", i, gamma[i]);
			}
		}
		for(int i=1;i<max_nCluster;i++)
		{
			printf("gamma[%d] = %.3f\n", i, gamma[i]);
		}
	}
}
void assign_labels(const int *nneigh, const int *ord_rho, 
									 const int *icl, float *cl, const int nClass, const int num) 
{
	for(int i=0;i<nClass;i++)
	{
		cl[icl[i]] = i+1;
	}
	for(int i=0;i<num;i++)
	{
		if(cl[ord_rho[i]] == -1)
		{
			cl[ord_rho[i]] = cl[nneigh[ord_rho[i]]];
		}
	}
}
static int cusalc_ME_LocalClustering(lua_State *L)
{
	// output: inds (d_inds)
	THCState *state = getCutorchState(L);
	THCudaTensor *dist = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *inds = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	long roi_num       = lua_tonumber(L, 4);
	float dc           = lua_tonumber(L, 5);
	float max_dist     = lua_tonumber(L, 6);
	long  cluster_flag = lua_tonumber(L, 7);
	long  max_nCluster = lua_tonumber(L, 8);

	float* d_dist = THCudaTensor_data(state, dist);
	float* d_inds = THCudaTensor_data(state, inds);
	
	// device params
	float* d_rho;
	float* d_delta;
	float* d_gamma;
	int*   d_nneigh;
	int*   d_ord_rho;
		
	//host params
	float *h_rho       = (float*)malloc(roi_num*sizeof(float));
	float *h_delta     = (float*)malloc(roi_num*sizeof(float));
	float *h_gamma     = (float*)malloc(roi_num*sizeof(float));
	float *h_inds      = (float*)malloc(roi_num*sizeof(float));
	int   *h_nneigh    = (int*)malloc(roi_num*sizeof(int));
	int   *h_ord_rho   = (int*)malloc(roi_num*sizeof(int));
	int   *h_ord_delta = (int*)malloc(roi_num*sizeof(int));
	int   *h_ord_gamma = (int*)malloc(roi_num*sizeof(int));
	int   *h_icl       = (int*)malloc(max_nCluster*sizeof(int));
	
	int nClass = 0;
	//init icl
	for(int i=0;i<max_nCluster;i++) h_icl[i] = -1;
	
	// malloc memory for device params
	cudaMalloc((void**)&d_rho,     roi_num*sizeof(float));
	cudaMalloc((void**)&d_delta,   roi_num*sizeof(float));
	cudaMalloc((void**)&d_gamma,   roi_num*sizeof(float));
	cudaMalloc((void**)&d_nneigh,  roi_num*sizeof(int));
	cudaMalloc((void**)&d_ord_rho, roi_num*sizeof(int));
	
	
	//compute rho
  compute_rho<<<GET_BLOCKS(roi_num), 1>>>(d_dist, d_rho, roi_num, dc);
	
	//sort rho using cpu (host)   !!!!! h_rho is not equal to d_rho because sorted
	cudaMemcpy(h_rho, d_rho, roi_num*sizeof(float), cudaMemcpyDeviceToHost);
	node_qsort_descend(h_rho, h_ord_rho, roi_num);
	cudaMemcpy(d_ord_rho, h_ord_rho, roi_num*sizeof(float), cudaMemcpyHostToDevice);
	
	
	//compute delta and nneigh
	compute_delta<<<GET_BLOCKS(roi_num), 1>>>(d_dist, d_ord_rho, d_delta, d_nneigh, roi_num, max_dist);	
	
	// compute gamma
  compute_gamma<<<GET_BLOCKS(roi_num), 1>>>(d_rho, d_delta, d_gamma, roi_num);
	
	//sort gamma and delta
	cudaMemcpy(h_delta, d_delta, roi_num*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_gamma, d_gamma, roi_num*sizeof(float), cudaMemcpyDeviceToHost);
	node_qsort_descend(h_delta, h_ord_delta, roi_num);
	node_qsort_descend(h_gamma, h_ord_gamma, roi_num);
	
	//compute icl
	compute_icl(h_ord_gamma, h_gamma, h_icl, &nClass, max_nCluster, cluster_flag);
	
	//assign inds
	cudaMemcpy(h_nneigh, d_nneigh, roi_num*sizeof(int  ), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_inds,   d_inds,   roi_num*sizeof(float), cudaMemcpyDeviceToHost);
  assign_labels(h_nneigh, h_ord_rho, h_icl, h_inds, nClass, roi_num);
	
	/*for(int i=0;i<roi_num;i++)
	{
		printf("inds[%d]=%.0f\n", i, h_inds[i]);
	}*/
	
	// output
	cudaMemcpy(d_inds, h_inds, roi_num*sizeof(float), cudaMemcpyHostToDevice);	
	
	cudaFree(d_rho);
	cudaFree(d_delta);
	cudaFree(d_gamma);
	cudaFree(d_nneigh);
	cudaFree(d_ord_rho);
	
	free(h_rho);
	free(h_delta);
	free(h_gamma);
	free(h_inds);
	free(h_nneigh);
	free(h_ord_rho);
	free(h_ord_delta);
	free(h_ord_gamma);
	free(h_icl);
	
	return 1;
}
static int cusalc_ME_LocalClusteringVis(lua_State *L)
{
	// output: inds (d_inds)
	THCState *state = getCutorchState(L);
	THCudaTensor *dist = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *inds = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	long roi_num       = lua_tonumber(L, 4);
	float dc           = lua_tonumber(L, 5);
	float max_dist     = lua_tonumber(L, 6);
	long  max_nCluster = lua_tonumber(L, 7);

	float* d_dist = THCudaTensor_data(state, dist);
	float* d_inds = THCudaTensor_data(state, inds);
	
	// device params
	float* d_rho;
	float* d_delta;
	float* d_gamma;
	int*   d_nneigh;
	int*   d_ord_rho;
		
	//host params
	float *h_rho       = (float*)malloc(roi_num*sizeof(float));
	float *h_delta     = (float*)malloc(roi_num*sizeof(float));
	float *h_gamma     = (float*)malloc(roi_num*sizeof(float));
	float *h_inds      = (float*)malloc(roi_num*sizeof(float));
	int   *h_nneigh    = (int*)malloc(roi_num*sizeof(int));
	int   *h_ord_rho   = (int*)malloc(roi_num*sizeof(int));
	int   *h_ord_delta = (int*)malloc(roi_num*sizeof(int));
	int   *h_ord_gamma = (int*)malloc(roi_num*sizeof(int));
	int   *h_icl       = (int*)malloc(max_nCluster*sizeof(int));
	
	int nClass = 0;
	//init icl
	for(int i=0;i<max_nCluster;i++) h_icl[i] = -1;
	
	// malloc memory for device params
	cudaMalloc((void**)&d_rho,     roi_num*sizeof(float));
	cudaMalloc((void**)&d_delta,   roi_num*sizeof(float));
	cudaMalloc((void**)&d_gamma,   roi_num*sizeof(float));
	cudaMalloc((void**)&d_nneigh,  roi_num*sizeof(int));
	cudaMalloc((void**)&d_ord_rho, roi_num*sizeof(int));
	
	
	//compute rho
  compute_rho<<<GET_BLOCKS(roi_num), 1>>>(d_dist, d_rho, roi_num, dc);
	
	//sort rho using cpu (host)   !!!!! h_rho is not equal to d_rho because sorted
	cudaMemcpy(h_rho, d_rho, roi_num*sizeof(float), cudaMemcpyDeviceToHost);
	node_qsort_descend(h_rho, h_ord_rho, roi_num);
	cudaMemcpy(d_ord_rho, h_ord_rho, roi_num*sizeof(float), cudaMemcpyHostToDevice);
	
	
	//compute delta and nneigh
	compute_delta<<<GET_BLOCKS(roi_num), 1>>>(d_dist, d_ord_rho, d_delta, d_nneigh, roi_num, max_dist);	
	
	// compute gamma
  compute_gamma<<<GET_BLOCKS(roi_num), 1>>>(d_rho, d_delta, d_gamma, roi_num);
	
	//sort gamma and delta
	cudaMemcpy(h_delta, d_delta, roi_num*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_gamma, d_gamma, roi_num*sizeof(float), cudaMemcpyDeviceToHost);
	node_qsort_descend(h_delta, h_ord_delta, roi_num);
	node_qsort_descend(h_gamma, h_ord_gamma, roi_num);
	
	//compute icl
	compute_icl_vis(h_ord_gamma, h_gamma, h_ord_delta, h_icl, &nClass, max_nCluster, 1);

	//assign inds
	cudaMemcpy(h_nneigh, d_nneigh, roi_num*sizeof(int  ), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_inds,   d_inds,   roi_num*sizeof(float), cudaMemcpyDeviceToHost);
  assign_labels(h_nneigh, h_ord_rho, h_icl, h_inds, nClass, roi_num);
	
	/*for(int i=0;i<roi_num;i++)
	{
		printf("inds[%d]=%.0f\n", i, h_inds[i]);
	}*/
	
	// output
	cudaMemcpy(d_inds, h_inds, roi_num*sizeof(float), cudaMemcpyHostToDevice);	
	
	cudaFree(d_rho);
	cudaFree(d_delta);
	cudaFree(d_gamma);
	cudaFree(d_nneigh);
	cudaFree(d_ord_rho);
	
	free(h_rho);
	free(h_delta);
	free(h_gamma);
	free(h_inds);
	free(h_nneigh);
	free(h_ord_rho);
	free(h_ord_delta);
	free(h_ord_gamma);
	free(h_icl);
	
	return 1;
}
//End: Local Clustering  //==================================================================	




template <typename Dtype>
__global__ void ComputeOverlap(
	const Dtype *rois, 
	const Dtype *gtroi, 
	Dtype *overlap, 
	const int ngtRoi,
	const int nRoi)
{
  CUDA_KERNEL_LOOP(idx, ngtRoi*nRoi)
  {
  	int i = idx / ngtRoi;
  	int j = idx % ngtRoi;
		Dtype xx1    = max(gtroi[5*j  ], rois[5*i  ]);
		Dtype yy1    = max(gtroi[5*j+1], rois[5*i+1]);
		Dtype xx2    = min(gtroi[5*j+2], rois[5*i+2]);
		Dtype yy2    = min(gtroi[5*j+3], rois[5*i+3]);
		Dtype w      = max(0.0, xx2-xx1+1);
		Dtype h      = max(0.0, yy2-yy1+1);

		Dtype inter_ = w*h;
		Dtype union_ =   (gtroi[5*j+2] - gtroi[5*j] +1) * (gtroi[5*j+3] - gtroi[5*j+1]+1)
								   + (rois[ 5*i+2] - rois[ 5*i]+1)  * (rois[ 5*i+3] - rois[ 5*i+1]+1)
								   - inter_;
		overlap[i*ngtRoi+j]   = inter_/union_;
  }
}
static int cusalc_ME_ComputeOverlap(lua_State *L)
{	
	//rois must be N*5 dims !!
	THCState *state          = getCutorchState(L);
	THCudaTensor *th_rois    = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *th_gtroi   = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *th_overlap = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	const long ngtRoi        = lua_tonumber(L, 5);
	const long nRoi          = lua_tonumber(L, 6);

	const float* rois    = THCudaTensor_data(state, th_rois);
	const float* gtroi   = THCudaTensor_data(state, th_gtroi);
	float* overlap = THCudaTensor_data(state, th_overlap);
	
  ComputeOverlap<<<GET_BLOCKS(ngtRoi*nRoi), CUDA_NUM_THREADS>>>(rois, gtroi, overlap, ngtRoi, nRoi);
	return 1;
}



template <typename Dtype>
__global__ void BoxGradBackward(
	const Dtype *gradOut, 
	Dtype *gradIn, 
	const Dtype *BoxIndex,
	const int roi_num, 
	const int cls_num)
{
  CUDA_KERNEL_LOOP(idx, roi_num*cls_num)
  {
  	int iRoi = idx / cls_num;
  	int iCls = idx % cls_num;
  	int iRoiOutput = BoxIndex[iRoi];

		gradIn[iRoiOutput*cls_num+iCls] = gradOut[iRoi*cls_num+iCls];
  }
}
static int cusalc_ME_BoxGradBackward(lua_State *L)
{	
	THCState *state = getCutorchState(L);
	THCudaTensor *th_gradIn   = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
	THCudaTensor *th_gradOut  = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
	THCudaTensor *th_BoxIndex = (THCudaTensor*)luaT_checkudata(L, 4, "torch.CudaTensor");
	const long roi_num        = lua_tonumber(L, 5);
	const long cls_num        = lua_tonumber(L, 6);

	const float* gradOut  = THCudaTensor_data(state, th_gradOut);
	const float* BoxIndex = THCudaTensor_data(state, th_BoxIndex);
	float* gradIn         = THCudaTensor_data(state, th_gradIn);
	
  BoxGradBackward<<<GET_BLOCKS(roi_num), CUDA_NUM_THREADS>>>(
  	gradOut,
  	gradIn,
  	BoxIndex,
  	roi_num,
  	cls_num);

	return 1;
}


static const struct luaL_Reg cusalc_ME__ [] = {
    {"ME_SplitInput",         cusalc_ME_SplitInput},
    {"ME_MergeInputFast",     cusalc_ME_MergeInputFast},
		{"ME_MergeGrads",         cusalc_ME_MergeGrads},
		{"ME_MergeGradsFast",     cusalc_ME_MergeGradsFast},
		{"ME_MergeScores",        cusalc_ME_MergeScores},
		{"ME_MergeScoresFast",    cusalc_ME_MergeScoresFast},
		{"ME_MergeScoresVis",     cusalc_ME_MergeScoresVis},
		{"ME_ComputeDistance",    cusalc_ME_ComputeDistance},
		{"ME_LocalClustering",    cusalc_ME_LocalClustering},
		{"ME_LocalClusteringVis", cusalc_ME_LocalClusteringVis},
		{"ME_ComputeOverlap",     cusalc_ME_ComputeOverlap},
		{"ME_BoxGradBackward",    cusalc_ME_BoxGradBackward},
    {NULL, NULL}
};

static void cusalc_ME_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.CudaTensor");
    luaT_registeratname(L, cusalc_ME__, "salc");
    lua_pop(L,1);
}


