# -*- cmake -*-

# USE_KDU can be set when launching cmake as an option using the argument -DUSE_KDU:BOOL=ON
# When building using proprietary binaries though (i.e. having access to LL private servers), 
# we always build with KDU
if (INSTALL_PROPRIETARY)
  set(USE_KDU ON CACHE BOOL "Use Kakadu library.")
endif (INSTALL_PROPRIETARY)

set( ND_KDU_SUFFIX "" )
if( ADDRESS_SIZE EQUAL 64 )
  if( WINDOWS OR LINUX )
    set( ND_KDU_SUFFIX "_x64" )
  endif( WINDOWS OR LINUX )
endif( ADDRESS_SIZE EQUAL 64 )
    
include_guard()
add_library( ll::kdu INTERFACE IMPORTED )

if (USE_KDU)
  include(Prebuilt)
  use_prebuilt_binary(kdu)
  if (WINDOWS)
    target_link_libraries( ll::kdu INTERFACE
      debug ${ARCH_PREBUILT_DIRS_DEBUG}/kdud.lib
      optimized ${ARCH_PREBUILT_DIRS_RELEASE}/kdu.lib
      )
  else (WINDOWS)
    target_link_libraries( ll::kdu INTERFACE libkdu.a)
  endif (WINDOWS)

  target_include_directories( ll::kdu SYSTEM INTERFACE
          ${LIBS_PREBUILT_DIR}/include/kdu
          )
endif (USE_KDU)
