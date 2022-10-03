# -*- cmake -*-
include(Prebuilt)

use_prebuilt_binary(jasper)

add_library( FV::jasper INTERFACE IMPORTED )
target_link_libraries( FV::jasper INTERFACE jasper )
target_include_directories( FV::jasper SYSTEM INTERFACE  ${LIBS_PREBUILT_DIR}/include/jasper)
