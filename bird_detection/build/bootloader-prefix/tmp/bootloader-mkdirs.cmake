# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/Users/usuario/esp/idf/esp-idf/components/bootloader/subproject"
  "/Users/usuario/esp/projects_tf/person_detection/build/bootloader"
  "/Users/usuario/esp/projects_tf/person_detection/build/bootloader-prefix"
  "/Users/usuario/esp/projects_tf/person_detection/build/bootloader-prefix/tmp"
  "/Users/usuario/esp/projects_tf/person_detection/build/bootloader-prefix/src/bootloader-stamp"
  "/Users/usuario/esp/projects_tf/person_detection/build/bootloader-prefix/src"
  "/Users/usuario/esp/projects_tf/person_detection/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/usuario/esp/projects_tf/person_detection/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/usuario/esp/projects_tf/person_detection/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
