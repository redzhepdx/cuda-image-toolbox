# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.3

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.3.0/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.3.0/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/htkibar/cuda-image-toolbox

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/htkibar/cuda-image-toolbox

# Include any dependencies generated for this target.
include CMakeFiles/DisplayImage.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/DisplayImage.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/DisplayImage.dir/flags.make

CMakeFiles/DisplayImage.dir/test.cpp.o: CMakeFiles/DisplayImage.dir/flags.make
CMakeFiles/DisplayImage.dir/test.cpp.o: test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/htkibar/cuda-image-toolbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/DisplayImage.dir/test.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/DisplayImage.dir/test.cpp.o -c /Users/htkibar/cuda-image-toolbox/test.cpp

CMakeFiles/DisplayImage.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/DisplayImage.dir/test.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/htkibar/cuda-image-toolbox/test.cpp > CMakeFiles/DisplayImage.dir/test.cpp.i

CMakeFiles/DisplayImage.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/DisplayImage.dir/test.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/htkibar/cuda-image-toolbox/test.cpp -o CMakeFiles/DisplayImage.dir/test.cpp.s

CMakeFiles/DisplayImage.dir/test.cpp.o.requires:

.PHONY : CMakeFiles/DisplayImage.dir/test.cpp.o.requires

CMakeFiles/DisplayImage.dir/test.cpp.o.provides: CMakeFiles/DisplayImage.dir/test.cpp.o.requires
	$(MAKE) -f CMakeFiles/DisplayImage.dir/build.make CMakeFiles/DisplayImage.dir/test.cpp.o.provides.build
.PHONY : CMakeFiles/DisplayImage.dir/test.cpp.o.provides

CMakeFiles/DisplayImage.dir/test.cpp.o.provides.build: CMakeFiles/DisplayImage.dir/test.cpp.o


# Object files for target DisplayImage
DisplayImage_OBJECTS = \
"CMakeFiles/DisplayImage.dir/test.cpp.o"

# External object files for target DisplayImage
DisplayImage_EXTERNAL_OBJECTS =

DisplayImage: CMakeFiles/DisplayImage.dir/test.cpp.o
DisplayImage: CMakeFiles/DisplayImage.dir/build.make
DisplayImage: /usr/local/lib/libopencv_videostab.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_ts.a
DisplayImage: /usr/local/lib/libopencv_superres.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_stitching.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_contrib.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_nonfree.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_ocl.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_gpu.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_photo.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_objdetect.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_legacy.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_video.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_ml.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_calib3d.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_features2d.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_highgui.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_imgproc.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_flann.2.4.11.dylib
DisplayImage: /usr/local/lib/libopencv_core.2.4.11.dylib
DisplayImage: CMakeFiles/DisplayImage.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/htkibar/cuda-image-toolbox/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable DisplayImage"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/DisplayImage.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/DisplayImage.dir/build: DisplayImage

.PHONY : CMakeFiles/DisplayImage.dir/build

CMakeFiles/DisplayImage.dir/requires: CMakeFiles/DisplayImage.dir/test.cpp.o.requires

.PHONY : CMakeFiles/DisplayImage.dir/requires

CMakeFiles/DisplayImage.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/DisplayImage.dir/cmake_clean.cmake
.PHONY : CMakeFiles/DisplayImage.dir/clean

CMakeFiles/DisplayImage.dir/depend:
	cd /Users/htkibar/cuda-image-toolbox && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/htkibar/cuda-image-toolbox /Users/htkibar/cuda-image-toolbox /Users/htkibar/cuda-image-toolbox /Users/htkibar/cuda-image-toolbox /Users/htkibar/cuda-image-toolbox/CMakeFiles/DisplayImage.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/DisplayImage.dir/depend

