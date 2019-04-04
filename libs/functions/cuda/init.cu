#include "luaT.h"
#include "THC.h"
#include "utils.h"
#include "ME.cu"


LUA_EXTERNC DLL_EXPORT int luaopen_libcusalc(lua_State *L);

int luaopen_libcusalc(lua_State *L)
{
  lua_newtable(L);

  cusalc_ME_init(L);
	
  return 1;
}
