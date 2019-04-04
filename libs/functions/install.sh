#!/bin/bash
now()  
{  
    date "+%Y-%m-%d %H:%M:%S"  
} 


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Install CPU version..."
cd $DIR/c
rm -r ./build
luarocks make salc-scm-1.rockspec
#echo "Testing CPU version..."
#th ./test.lua

echo "Install GPU version..."
cd $DIR/cuda
rm -r ./build
luarocks make cusalc-scm-1.rockspec
#echo "Testing GPU version..."
#th ./test.lua

echo "-------------------------------"
echo "All done!  " $(now)

