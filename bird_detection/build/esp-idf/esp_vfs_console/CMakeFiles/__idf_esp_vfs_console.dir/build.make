# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.27.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.27.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/usuario/esp/projects_tf/person_detection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/usuario/esp/projects_tf/person_detection/build

# Include any dependencies generated for this target.
include esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/compiler_depend.make

# Include the progress variables for this target.
include esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/progress.make

# Include the compile flags for this target's objects.
include esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/flags.make

esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.obj: esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/flags.make
esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.obj: /Users/usuario/esp/idf/esp-idf/components/esp_vfs_console/vfs_console.c
esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.obj: esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/usuario/esp/projects_tf/person_detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.obj"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/esp_vfs_console && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.obj -MF CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.obj.d -o CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.obj -c /Users/usuario/esp/idf/esp-idf/components/esp_vfs_console/vfs_console.c

esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.i"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/esp_vfs_console && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/usuario/esp/idf/esp-idf/components/esp_vfs_console/vfs_console.c > CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.i

esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.s"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/esp_vfs_console && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/usuario/esp/idf/esp-idf/components/esp_vfs_console/vfs_console.c -o CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.s

# Object files for target __idf_esp_vfs_console
__idf_esp_vfs_console_OBJECTS = \
"CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.obj"

# External object files for target __idf_esp_vfs_console
__idf_esp_vfs_console_EXTERNAL_OBJECTS =

esp-idf/esp_vfs_console/libesp_vfs_console.a: esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/vfs_console.c.obj
esp-idf/esp_vfs_console/libesp_vfs_console.a: esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/build.make
esp-idf/esp_vfs_console/libesp_vfs_console.a: esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/usuario/esp/projects_tf/person_detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library libesp_vfs_console.a"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/esp_vfs_console && $(CMAKE_COMMAND) -P CMakeFiles/__idf_esp_vfs_console.dir/cmake_clean_target.cmake
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/esp_vfs_console && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/__idf_esp_vfs_console.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/build: esp-idf/esp_vfs_console/libesp_vfs_console.a
.PHONY : esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/build

esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/clean:
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/esp_vfs_console && $(CMAKE_COMMAND) -P CMakeFiles/__idf_esp_vfs_console.dir/cmake_clean.cmake
.PHONY : esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/clean

esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/depend:
	cd /Users/usuario/esp/projects_tf/person_detection/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/usuario/esp/projects_tf/person_detection /Users/usuario/esp/idf/esp-idf/components/esp_vfs_console /Users/usuario/esp/projects_tf/person_detection/build /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/esp_vfs_console /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : esp-idf/esp_vfs_console/CMakeFiles/__idf_esp_vfs_console.dir/depend

