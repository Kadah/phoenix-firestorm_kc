#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate sl

export AUTOBUILD_VARIABLES_FILE=$HOME/Dev/sl/fs-build-variables/variables

autobuild installables edit fmodstudio platform=linux64 hash=f9bb7110e3ef4e7b50f41f1d54ba1d99 url=file:////opt/sl_3p/fmodstudio-2.02.09-linux64-222940517.tar.bz2

CUDACXX=nvcc

time autobuild configure -A 64 -c ReleaseFS_open -- -DPACKAGE:BOOL=On --chan="KC_CUDA" --fmodstudio  -DUSE_NVJPEG2K:BOOL=On -DUSE_SDL2:BOOL=On --clean

time autobuild build -A 64 -c ReleaseFS_open -- -DPACKAGE:BOOL=On --chan="KC_CUDA" --fmodstudio  -DUSE_NVJPEG2K:BOOL=On -DUSE_SDL2:BOOL=On
