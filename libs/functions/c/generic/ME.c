#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ME.c"
#else

#include <assert.h>


/* define functions*/
static int salc_(ME_SplitInput)(lua_State *L)
{
  printf("Do not support CPU, please use GPU mode !!!!");
	exit(-1);
}

static int salc_(ME_LocalConsistency)(lua_State *L)
{
  THTensor *th_dist   = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *th_inds   = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *th_clique = luaT_checkudata(L, 4, torch_Tensor);
  const long roi_num        = lua_tonumber(L, 5);
  const float ov_thresh     = lua_tonumber(L, 6);
	
	const real *dist   = THTensor_(data)(th_dist);
	real *inds   = THTensor_(data)(th_inds);
	real *clique = THTensor_(data)(th_clique);
	
	int i,j,k, nCliqElem;
	int ID = 1; /*inds start from 1*/
	int flag_in_clique;
	int end_num;
	
	for(i=0;i<roi_num;i++)
	{
		/*if inds[i] is not assigned*/
		if(inds[i] == 0)
		{
			/*init clique*/
			for(j=0;j<200;j++)
			{
				clique[j]=-1;
			}
			
			/*compute clique and assign ID*/
			inds[i]   = ID;
			clique[0] = i;
			nCliqElem = 1;
			end_num = i+40 > roi_num ? roi_num : i+40;
			for(j=i+1;j<end_num;j++)
			{
				/*check roi j is in clique or not*/
				flag_in_clique = 1;
				for(k=0;k<nCliqElem;k++)
				{
					if(dist[j*roi_num+(int)clique[k]] < ov_thresh) {flag_in_clique=0;break;}
				}
				
				/*if roi j in clique*/
				if(flag_in_clique)
				{
					inds[j] = ID;
					clique[nCliqElem] = j;
					nCliqElem += 1;
				}
			}
			ID += 1;
		}
	}
}


static int salc_(ME_AssignInds)(lua_State *L)
{
  THTensor *th_inds_all = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *th_inds     = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *th_mask     = luaT_checkudata(L, 4, torch_Tensor);
  long nAllRois         = lua_tonumber(L, 5);
  long nTopLocal        = lua_tonumber(L, 6);
  long nTopClass        = lua_tonumber(L, 7);
	
	real *inds_all = THTensor_(data)(th_inds_all);
	real *inds     = THTensor_(data)(th_inds);
	real *mask     = THTensor_(data)(th_mask);
	
	int i;
	int nt = 0;
	int nn = 0;
	for(i=0;i<nAllRois;i++)
	{
		if(mask[i]==1)
		{
			inds_all[i*2  ] = inds[nt*2  ];
			inds_all[i*2+1] = inds[nt*2+1];
			nt += 1;
		}
		else
		{
			inds_all[i*2  ] = nTopLocal+1;
			inds_all[i*2+1] = nn%nTopClass + 1;
			nTopLocal +=1;
			nn += 1;
		}
	}
}

static int salc_(ME_AssignIndsFast)(lua_State *L)
{
  THTensor *th_inds_all = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *th_inds     = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *th_mask     = luaT_checkudata(L, 4, torch_Tensor);
  long nAllRois         = lua_tonumber(L, 5);
  long nTopLocal        = lua_tonumber(L, 6);
	
	real *inds_all = THTensor_(data)(th_inds_all);
	real *inds     = THTensor_(data)(th_inds);
	real *mask     = THTensor_(data)(th_mask);
	
	int i;
	int nt = 0;
	int nn = 0;
	int temp_for_assert = nTopLocal;
	for(i=0;i<nAllRois;i++)
	{
		if(mask[i]==1)
		{
			inds_all[i] = inds[nt];
			nt++;
		}
		else
		{
			inds_all[i] = nTopLocal+1;
			nTopLocal++;
		}
	}
	assert(temp_for_assert == nt);
}

static int salc_(ME_TransInds)(lua_State *L)
{
  THTensor *th_out_inds = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *th_inds     = luaT_checkudata(L, 3, torch_Tensor);
  long num              = lua_tonumber(L, 4);
  long nClique          = lua_tonumber(L, 5);
  long c_i              = lua_tonumber(L, 6);
	
	real *out_inds = THTensor_(data)(th_out_inds);
	real *inds     = THTensor_(data)(th_inds);
	
	int i,nn;
	nn=0;
	for(i=0;i<num;i++)
	{
		if(inds[i*2+1] == c_i)
		{
			out_inds[nn *3    ] = inds[i*2];
			out_inds[nn *3 + 1] = 0;
			out_inds[nn *3 + 2] = i+1;
			nn +=1;
		}
	}
	assert(nn == nClique);
}

static int salc_(ME_TransIndsFast)(lua_State *L)
{
  THTensor *th_out_inds = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *th_inds     = luaT_checkudata(L, 3, torch_Tensor);
  long num              = lua_tonumber(L, 4);
	
	real *out_inds = THTensor_(data)(th_out_inds);
	real *inds     = THTensor_(data)(th_inds);
	
	int i,nn;
	nn=0;
	for(i=0;i<num;i++)
	{
		out_inds[nn *3    ] = inds[i];
		out_inds[nn *3 + 1] = 0;
		out_inds[nn *3 + 2] = i+1;
		nn +=1;
	}
	assert(nn == num);
}

static int salc_(ME_ReTransInds)(lua_State *L)
{
  THTensor *th_inds      = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *th_temp      = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *th_temp_val  = luaT_checkudata(L, 4, torch_Tensor);
  THTensor *th_temp_inds = luaT_checkudata(L, 5, torch_Tensor);
  long num               = lua_tonumber(L, 6);
	
	real *inds            = THTensor_(data)(th_inds);
	const real *temp      = THTensor_(data)(th_temp);
	const real *temp_val  = THTensor_(data)(th_temp_val);
	const real *temp_inds = THTensor_(data)(th_temp_inds);
	int i;
	int nn = 1;
	
	/* retranslate inds*/
	for(i=0;i<num;i++)
	{
		assert(nn<=temp_val[i]);
		inds[i*3  ] = temp_val[i];
		inds[i*3+1] = nn;
		inds[i*3+2] = temp[(int)temp_inds[i]-1];
		if(i<num-1 && temp_val[i]!=temp_val[i+1])
		{
			nn+=1;
		}
	}
}

static int salc_(ME_Indsli2gl)(lua_State *L)
{
  THTensor *th_inds      = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *th_mask_top  = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *th_mask_all  = luaT_checkudata(L, 4, torch_Tensor);
  long nRoiClsi          = lua_tonumber(L, 5);
  long nAllRoi           = lua_tonumber(L, 6);
	
	real *inds            = THTensor_(data)(th_inds);
	const real *mask_top  = THTensor_(data)(th_mask_top);
	const real *mask_all  = THTensor_(data)(th_mask_all);
	int i;
	int nn = 0;
	
	/* retranslate inds*/
	for(i=0;i<nAllRoi;i++)
	{
		if(mask_top[i] == 1 || mask_all[i] == 0)
		{
			inds[nn] = i+1;
			nn += 1;
		}
	}
	assert(nn == nRoiClsi);
}




/* regist function in modual*/
static const struct luaL_Reg salc_(ME__) [] = {
    {"ME_SplitInput", salc_(ME_SplitInput)},
    {"ME_LocalConsistency", salc_(ME_LocalConsistency)},
    {"ME_AssignInds", salc_(ME_AssignInds)},
    {"ME_AssignIndsFast", salc_(ME_AssignIndsFast)},
    {"ME_TransInds", salc_(ME_TransInds)},
    {"ME_TransIndsFast", salc_(ME_TransIndsFast)},
    {"ME_ReTransInds", salc_(ME_ReTransInds)},
    {"ME_Indsli2gl", salc_(ME_Indsli2gl)},
    {NULL,NULL}
};

static void salc_(ME_init)(lua_State *L)
{
    luaT_pushmetatable(L, torch_Tensor);
    luaT_registeratname(L, salc_(ME__), "salc");
    lua_pop(L,1);
}

#endif
