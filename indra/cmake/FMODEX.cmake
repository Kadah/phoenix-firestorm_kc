# -*- cmake -*-

# FMODEX can be set when launching the make using the argument -DFMOD:BOOL=ON
# No longer used by default, see FMODSTRUDIO.
# Open source devs should use the -DFMODEX:BOOL=ON then if they want to build with FMODEX, whether
# they are using USESYSTEMLIBS or not.

if (FMODEX)
  if (USESYSTEMLIBS)
    # In that case, we use the version of the library installed on the system
    set(FMODEX_FIND_REQUIRED ON)
    include(FindFMODEX)
  else (USESYSTEMLIBS)
    if (FMODEX_LIBRARY AND FMODEX_INCLUDE_DIR)
      # If the path have been specified in the arguments, use that
      set(FMODEX_LIBRARIES ${FMODEX_LIBRARY})
      MESSAGE(STATUS "Using FMODEX path: ${FMODEX_LIBRARIES}, ${FMODEX_INCLUDE_DIR}")
    else (FMODEX_LIBRARY AND FMODEX_INCLUDE_DIR)
      # If not, we're going to try to get the package listed in autobuild.xml
      # Note: if you're not using INSTALL_PROPRIETARY, the package URL should be local (file:/// URL) 
      # as accessing the private LL location will fail if you don't have the credential
      include(Prebuilt)
      use_prebuilt_binary(fmodex)    
      if (WINDOWS)
        set(FMODEX_LIBRARY 
            debug fmodexL_vc
            optimized fmodex_vc)
      elseif (DARWIN)
        set(FMODEX_LIBRARY 
            debug fmodexL
            optimized fmodex)
      elseif (LINUX)
        set(FMODEX_LIBRARY 
            debug fmodexL
            optimized fmodex)
      endif (WINDOWS)
      set(FMODEX_LIBRARIES ${FMODEX_LIBRARY})
      set(FMODEX_INCLUDE_DIR ${LIBS_PREBUILT_DIR}/include/fmodex)
    endif (FMODEX_LIBRARY AND FMODEX_INCLUDE_DIR)
  endif (USESYSTEMLIBS)
endif (FMODEX)

