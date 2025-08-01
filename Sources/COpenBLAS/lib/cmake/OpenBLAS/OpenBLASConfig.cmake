# OpenBLASConfig.cmake
# --------------------
#
# OpenBLAS cmake module.
# This module sets the following variables in your project::
#
#   OpenBLAS_FOUND - true if OpenBLAS and all required components found on the system
#   OpenBLAS_VERSION - OpenBLAS version in format Major.Minor.Release
#   OpenBLAS_INCLUDE_DIRS - Directory where OpenBLAS header is located.
#   OpenBLAS_INCLUDE_DIR - same as DIRS
#   OpenBLAS_LIBRARIES - OpenBLAS library to link against.
#   OpenBLAS_LIBRARY - same as LIBRARIES
#
#
# Available components::
#
##   shared - search for only shared library
##   static - search for only static library
#   serial - search for unthreaded library
#   pthread - search for native pthread threaded library
#   openmp - search for OpenMP threaded library
#
#
# Exported targets::
#
# If OpenBLAS is found, this module defines the following :prop_tgt:`IMPORTED`
## target. Target is shared _or_ static, so, for both, use separate, not
## overlapping, installations. ::
#
#   OpenBLAS::OpenBLAS - the main OpenBLAS library #with header & defs attached.
#
#
# Suggested usage::
#
#   find_package(OpenBLAS)
#   find_package(OpenBLAS 0.2.20 EXACT CONFIG REQUIRED COMPONENTS pthread)
#
#
# The following variables can be set to guide the search for this package::
#
#   OpenBLAS_DIR - CMake variable, set to directory containing this Config file
#   CMAKE_PREFIX_PATH - CMake variable, set to root directory of this package
#   PATH - environment variable, set to bin directory of this package
#   CMAKE_DISABLE_FIND_PACKAGE_OpenBLAS - CMake variable, disables
#     find_package(OpenBLAS) when not REQUIRED, perhaps to force internal build


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was OpenBLASConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(PN OpenBLAS)

# need to check that the @USE_*@ evaluate to something cmake can perform boolean logic upon
if(OFF)
    set(${PN}_openmp_FOUND 1)
elseif(OFF)
    set(${PN}_pthread_FOUND 1)
else()
    set(${PN}_serial_FOUND 1)
endif()

check_required_components(${PN})

#-----------------------------------------------------------------------------
# Don't include targets if this file is being picked up by another
# project which has already built this as a subproject
#-----------------------------------------------------------------------------
if(NOT TARGET ${PN}::OpenBLAS)
    include("${CMAKE_CURRENT_LIST_DIR}/${PN}Targets.cmake")

    get_property(_loc TARGET ${PN}::OpenBLAS PROPERTY LOCATION)
    set(${PN}_LIBRARY ${_loc})
    get_property(_ill TARGET ${PN}::OpenBLAS PROPERTY INTERFACE_LINK_LIBRARIES)
    set(${PN}_LIBRARIES ${_ill})

    get_property(_id TARGET ${PN}::OpenBLAS PROPERTY INCLUDE_DIRECTORIES)
    set(${PN}_INCLUDE_DIR ${_id})
    get_property(_iid TARGET ${PN}::OpenBLAS PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    set(${PN}_INCLUDE_DIRS ${_iid})
endif()

