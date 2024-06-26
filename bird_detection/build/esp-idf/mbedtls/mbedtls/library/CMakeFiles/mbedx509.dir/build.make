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
include esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/compiler_depend.make

# Include the progress variables for this target.
include esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/progress.make

# Include the compile flags for this target's objects.
include esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/flags.make

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/pkcs7.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/flags.make
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/pkcs7.c.obj: /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/pkcs7.c
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/pkcs7.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/usuario/esp/projects_tf/person_detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/pkcs7.c.obj"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/pkcs7.c.obj -MF CMakeFiles/mbedx509.dir/pkcs7.c.obj.d -o CMakeFiles/mbedx509.dir/pkcs7.c.obj -c /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/pkcs7.c

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/pkcs7.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/mbedx509.dir/pkcs7.c.i"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/pkcs7.c > CMakeFiles/mbedx509.dir/pkcs7.c.i

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/pkcs7.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/mbedx509.dir/pkcs7.c.s"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/pkcs7.c -o CMakeFiles/mbedx509.dir/pkcs7.c.s

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/flags.make
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509.c.obj: /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509.c
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/usuario/esp/projects_tf/person_detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509.c.obj"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509.c.obj -MF CMakeFiles/mbedx509.dir/x509.c.obj.d -o CMakeFiles/mbedx509.dir/x509.c.obj -c /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509.c

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/mbedx509.dir/x509.c.i"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509.c > CMakeFiles/mbedx509.dir/x509.c.i

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/mbedx509.dir/x509.c.s"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509.c -o CMakeFiles/mbedx509.dir/x509.c.s

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_create.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/flags.make
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_create.c.obj: /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_create.c
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_create.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/usuario/esp/projects_tf/person_detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_create.c.obj"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_create.c.obj -MF CMakeFiles/mbedx509.dir/x509_create.c.obj.d -o CMakeFiles/mbedx509.dir/x509_create.c.obj -c /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_create.c

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_create.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/mbedx509.dir/x509_create.c.i"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_create.c > CMakeFiles/mbedx509.dir/x509_create.c.i

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_create.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/mbedx509.dir/x509_create.c.s"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_create.c -o CMakeFiles/mbedx509.dir/x509_create.c.s

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crl.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/flags.make
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crl.c.obj: /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_crl.c
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crl.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/usuario/esp/projects_tf/person_detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crl.c.obj"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crl.c.obj -MF CMakeFiles/mbedx509.dir/x509_crl.c.obj.d -o CMakeFiles/mbedx509.dir/x509_crl.c.obj -c /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_crl.c

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crl.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/mbedx509.dir/x509_crl.c.i"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_crl.c > CMakeFiles/mbedx509.dir/x509_crl.c.i

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crl.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/mbedx509.dir/x509_crl.c.s"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_crl.c -o CMakeFiles/mbedx509.dir/x509_crl.c.s

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crt.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/flags.make
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crt.c.obj: /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_crt.c
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crt.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/usuario/esp/projects_tf/person_detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crt.c.obj"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crt.c.obj -MF CMakeFiles/mbedx509.dir/x509_crt.c.obj.d -o CMakeFiles/mbedx509.dir/x509_crt.c.obj -c /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_crt.c

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crt.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/mbedx509.dir/x509_crt.c.i"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_crt.c > CMakeFiles/mbedx509.dir/x509_crt.c.i

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crt.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/mbedx509.dir/x509_crt.c.s"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_crt.c -o CMakeFiles/mbedx509.dir/x509_crt.c.s

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_csr.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/flags.make
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_csr.c.obj: /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_csr.c
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_csr.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/usuario/esp/projects_tf/person_detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_csr.c.obj"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_csr.c.obj -MF CMakeFiles/mbedx509.dir/x509_csr.c.obj.d -o CMakeFiles/mbedx509.dir/x509_csr.c.obj -c /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_csr.c

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_csr.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/mbedx509.dir/x509_csr.c.i"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_csr.c > CMakeFiles/mbedx509.dir/x509_csr.c.i

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_csr.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/mbedx509.dir/x509_csr.c.s"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509_csr.c -o CMakeFiles/mbedx509.dir/x509_csr.c.s

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/flags.make
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write.c.obj: /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509write.c
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/usuario/esp/projects_tf/person_detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write.c.obj"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write.c.obj -MF CMakeFiles/mbedx509.dir/x509write.c.obj.d -o CMakeFiles/mbedx509.dir/x509write.c.obj -c /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509write.c

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/mbedx509.dir/x509write.c.i"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509write.c > CMakeFiles/mbedx509.dir/x509write.c.i

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/mbedx509.dir/x509write.c.s"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509write.c -o CMakeFiles/mbedx509.dir/x509write.c.s

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_crt.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/flags.make
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_crt.c.obj: /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509write_crt.c
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_crt.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/usuario/esp/projects_tf/person_detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_crt.c.obj"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_crt.c.obj -MF CMakeFiles/mbedx509.dir/x509write_crt.c.obj.d -o CMakeFiles/mbedx509.dir/x509write_crt.c.obj -c /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509write_crt.c

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_crt.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/mbedx509.dir/x509write_crt.c.i"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509write_crt.c > CMakeFiles/mbedx509.dir/x509write_crt.c.i

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_crt.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/mbedx509.dir/x509write_crt.c.s"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509write_crt.c -o CMakeFiles/mbedx509.dir/x509write_crt.c.s

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_csr.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/flags.make
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_csr.c.obj: /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509write_csr.c
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_csr.c.obj: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/usuario/esp/projects_tf/person_detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building C object esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_csr.c.obj"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_csr.c.obj -MF CMakeFiles/mbedx509.dir/x509write_csr.c.obj.d -o CMakeFiles/mbedx509.dir/x509write_csr.c.obj -c /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509write_csr.c

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_csr.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/mbedx509.dir/x509write_csr.c.i"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509write_csr.c > CMakeFiles/mbedx509.dir/x509write_csr.c.i

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_csr.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/mbedx509.dir/x509write_csr.c.s"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library/x509write_csr.c -o CMakeFiles/mbedx509.dir/x509write_csr.c.s

# Object files for target mbedx509
mbedx509_OBJECTS = \
"CMakeFiles/mbedx509.dir/pkcs7.c.obj" \
"CMakeFiles/mbedx509.dir/x509.c.obj" \
"CMakeFiles/mbedx509.dir/x509_create.c.obj" \
"CMakeFiles/mbedx509.dir/x509_crl.c.obj" \
"CMakeFiles/mbedx509.dir/x509_crt.c.obj" \
"CMakeFiles/mbedx509.dir/x509_csr.c.obj" \
"CMakeFiles/mbedx509.dir/x509write.c.obj" \
"CMakeFiles/mbedx509.dir/x509write_crt.c.obj" \
"CMakeFiles/mbedx509.dir/x509write_csr.c.obj"

# External object files for target mbedx509
mbedx509_EXTERNAL_OBJECTS =

esp-idf/mbedtls/mbedtls/library/libmbedx509.a: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/pkcs7.c.obj
esp-idf/mbedtls/mbedtls/library/libmbedx509.a: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509.c.obj
esp-idf/mbedtls/mbedtls/library/libmbedx509.a: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_create.c.obj
esp-idf/mbedtls/mbedtls/library/libmbedx509.a: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crl.c.obj
esp-idf/mbedtls/mbedtls/library/libmbedx509.a: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_crt.c.obj
esp-idf/mbedtls/mbedtls/library/libmbedx509.a: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509_csr.c.obj
esp-idf/mbedtls/mbedtls/library/libmbedx509.a: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write.c.obj
esp-idf/mbedtls/mbedtls/library/libmbedx509.a: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_crt.c.obj
esp-idf/mbedtls/mbedtls/library/libmbedx509.a: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/x509write_csr.c.obj
esp-idf/mbedtls/mbedtls/library/libmbedx509.a: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/build.make
esp-idf/mbedtls/mbedtls/library/libmbedx509.a: esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/usuario/esp/projects_tf/person_detection/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX static library libmbedx509.a"
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && $(CMAKE_COMMAND) -P CMakeFiles/mbedx509.dir/cmake_clean_target.cmake
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mbedx509.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/build: esp-idf/mbedtls/mbedtls/library/libmbedx509.a
.PHONY : esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/build

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/clean:
	cd /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library && $(CMAKE_COMMAND) -P CMakeFiles/mbedx509.dir/cmake_clean.cmake
.PHONY : esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/clean

esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/depend:
	cd /Users/usuario/esp/projects_tf/person_detection/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/usuario/esp/projects_tf/person_detection /Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library /Users/usuario/esp/projects_tf/person_detection/build /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library /Users/usuario/esp/projects_tf/person_detection/build/esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : esp-idf/mbedtls/mbedtls/library/CMakeFiles/mbedx509.dir/depend

