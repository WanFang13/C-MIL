#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define salc_(NAME) TH_CONCAT_3(salc_, Real, NAME)

/* add include here*/
#include "generic/ME.c"
#include "THGenerateFloatTypes.h"


LUA_EXTERNC DLL_EXPORT int luaopen_libsalc(lua_State *L);

int luaopen_libsalc(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "salc");

	/* add init function here*/
  salc_FloatME_init(L);
  salc_DoubleME_init(L);

  return 1;
}
