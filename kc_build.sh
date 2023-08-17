#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate sl

export AUTOBUILD_VARIABLES_FILE=$HOME/Dev/sl/fs-build-variables/variables

autobuild installables edit fmodstudio platform=linux64 hash=5734ad11b2c5079366e059797e09f57d url=file:////opt/sl_3p/fmodstudio-2.02.15-linux64-232270400.tar.bz2
autobuild installables edit kdu platform=linux64 hash=501679bc25dcd5f45719af3754ae91ea url=file:////opt/sl_3p/kdu-linux64.tar.bz2

time autobuild configure -A 64 -c ReleaseFS -- -DPACKAGE:BOOL=On --chan="KC" --clean

time autobuild build -A 64 -c ReleaseFS -- -DPACKAGE:BOOL=On --chan="KC"
