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

# Utility rule file for efuse_test_table.

# Include any custom commands dependencies for this target.
include esp-idf/efuse/CMakeFiles/efuse_test_table.dir/compiler_depend.make

# Include the progress variables for this target.
include esp-idf/efuse/CMakeFiles/efuse_test_table.dir/progress.make

esp-idf/efuse/CMakeFiles/efuse_test_table:
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/efuse && /Users/usuario/esp/idf-tools/python_env/idf5.3_py3.11_env/bin/python /Users/usuario/esp/idf/esp-idf/components/efuse/efuse_table_gen.py /Users/usuario/esp/idf/esp-idf/components/efuse/test/esp_efuse_test_table.csv -t esp32 --max_blk_len 192

efuse_test_table: esp-idf/efuse/CMakeFiles/efuse_test_table
efuse_test_table: esp-idf/efuse/CMakeFiles/efuse_test_table.dir/build.make
.PHONY : efuse_test_table

# Rule to build all files generated by this target.
esp-idf/efuse/CMakeFiles/efuse_test_table.dir/build: efuse_test_table
.PHONY : esp-idf/efuse/CMakeFiles/efuse_test_table.dir/build

esp-idf/efuse/CMakeFiles/efuse_test_table.dir/clean:
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/efuse && $(CMAKE_COMMAND) -P CMakeFiles/efuse_test_table.dir/cmake_clean.cmake
.PHONY : esp-idf/efuse/CMakeFiles/efuse_test_table.dir/clean

esp-idf/efuse/CMakeFiles/efuse_test_table.dir/depend:
	cd /Users/usuario/esp/projects_tf/person_detection/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/usuario/esp/projects_tf/person_detection /Users/usuario/esp/idf/esp-idf/components/efuse /Users/usuario/esp/projects_tf/person_detection/build /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/efuse /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/efuse/CMakeFiles/efuse_test_table.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : esp-idf/efuse/CMakeFiles/efuse_test_table.dir/depend

