# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# compile C with /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc
C_DEFINES = -DESP_PLATFORM -DIDF_VER=\"v5.3-dev-2672-g6b9c2fca79\" -DMBEDTLS_CONFIG_FILE=\"mbedtls/esp_config.h\" -DSOC_MMU_PAGE_SIZE=CONFIG_MMU_PAGE_SIZE -DSOC_XTAL_FREQ_MHZ=CONFIG_XTAL_FREQ -D_GLIBCXX_HAVE_POSIX_SEMAPHORE -D_GLIBCXX_USE_POSIX_SEMAPHORE -D_GNU_SOURCE -D_POSIX_READER_WRITER_LOCKS

C_INCLUDES = -I/Users/usuario/esp/projects_tf/person_detection/build/config -I/Users/usuario/esp/idf/esp-idf/components/protocomm/include/common -I/Users/usuario/esp/idf/esp-idf/components/protocomm/include/security -I/Users/usuario/esp/idf/esp-idf/components/protocomm/include/transports -I/Users/usuario/esp/idf/esp-idf/components/protocomm/include/crypto/srp6a -I/Users/usuario/esp/idf/esp-idf/components/protocomm/proto-c -I/Users/usuario/esp/idf/esp-idf/components/protocomm/src/common -I/Users/usuario/esp/idf/esp-idf/components/newlib/platform_include -I/Users/usuario/esp/idf/esp-idf/components/freertos/config/include -I/Users/usuario/esp/idf/esp-idf/components/freertos/config/include/freertos -I/Users/usuario/esp/idf/esp-idf/components/freertos/config/xtensa/include -I/Users/usuario/esp/idf/esp-idf/components/freertos/FreeRTOS-Kernel/include -I/Users/usuario/esp/idf/esp-idf/components/freertos/FreeRTOS-Kernel/portable/xtensa/include -I/Users/usuario/esp/idf/esp-idf/components/freertos/FreeRTOS-Kernel/portable/xtensa/include/freertos -I/Users/usuario/esp/idf/esp-idf/components/freertos/esp_additions/include -I/Users/usuario/esp/idf/esp-idf/components/esp_hw_support/include -I/Users/usuario/esp/idf/esp-idf/components/esp_hw_support/include/soc -I/Users/usuario/esp/idf/esp-idf/components/esp_hw_support/include/soc/esp32 -I/Users/usuario/esp/idf/esp-idf/components/esp_hw_support/dma/include -I/Users/usuario/esp/idf/esp-idf/components/esp_hw_support/port/esp32/. -I/Users/usuario/esp/idf/esp-idf/components/esp_hw_support/port/esp32/include -I/Users/usuario/esp/idf/esp-idf/components/heap/include -I/Users/usuario/esp/idf/esp-idf/components/log/include -I/Users/usuario/esp/idf/esp-idf/components/soc/include -I/Users/usuario/esp/idf/esp-idf/components/soc/esp32 -I/Users/usuario/esp/idf/esp-idf/components/soc/esp32/include -I/Users/usuario/esp/idf/esp-idf/components/hal/platform_port/include -I/Users/usuario/esp/idf/esp-idf/components/hal/esp32/include -I/Users/usuario/esp/idf/esp-idf/components/hal/include -I/Users/usuario/esp/idf/esp-idf/components/esp_rom/include -I/Users/usuario/esp/idf/esp-idf/components/esp_rom/include/esp32 -I/Users/usuario/esp/idf/esp-idf/components/esp_rom/esp32 -I/Users/usuario/esp/idf/esp-idf/components/esp_common/include -I/Users/usuario/esp/idf/esp-idf/components/esp_system/include -I/Users/usuario/esp/idf/esp-idf/components/esp_system/port/soc -I/Users/usuario/esp/idf/esp-idf/components/esp_system/port/include/private -I/Users/usuario/esp/idf/esp-idf/components/xtensa/esp32/include -I/Users/usuario/esp/idf/esp-idf/components/xtensa/include -I/Users/usuario/esp/idf/esp-idf/components/xtensa/deprecated_include -I/Users/usuario/esp/idf/esp-idf/components/esp_timer/include -I/Users/usuario/esp/idf/esp-idf/components/lwip/include -I/Users/usuario/esp/idf/esp-idf/components/lwip/include/apps -I/Users/usuario/esp/idf/esp-idf/components/lwip/include/apps/sntp -I/Users/usuario/esp/idf/esp-idf/components/lwip/lwip/src/include -I/Users/usuario/esp/idf/esp-idf/components/lwip/port/include -I/Users/usuario/esp/idf/esp-idf/components/lwip/port/freertos/include -I/Users/usuario/esp/idf/esp-idf/components/lwip/port/esp32xx/include -I/Users/usuario/esp/idf/esp-idf/components/lwip/port/esp32xx/include/arch -I/Users/usuario/esp/idf/esp-idf/components/lwip/port/esp32xx/include/sys -I/Users/usuario/esp/idf/esp-idf/components/esp_wifi/include -I/Users/usuario/esp/idf/esp-idf/components/esp_wifi/wifi_apps/include -I/Users/usuario/esp/idf/esp-idf/components/esp_wifi/include/local -I/Users/usuario/esp/idf/esp-idf/components/esp_event/include -I/Users/usuario/esp/idf/esp-idf/components/esp_phy/include -I/Users/usuario/esp/idf/esp-idf/components/esp_phy/esp32/include -I/Users/usuario/esp/idf/esp-idf/components/esp_netif/include -I/Users/usuario/esp/idf/esp-idf/components/protobuf-c/protobuf-c -I/Users/usuario/esp/idf/esp-idf/components/mbedtls/port/include -I/Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/include -I/Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library -I/Users/usuario/esp/idf/esp-idf/components/mbedtls/esp_crt_bundle/include -I/Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/3rdparty/everest/include -I/Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/3rdparty/p256-m -I/Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/3rdparty/p256-m/p256-m -I/Users/usuario/esp/idf/esp-idf/components/console -I/Users/usuario/esp/idf/esp-idf/components/vfs/include -I/Users/usuario/esp/idf/esp-idf/components/esp_vfs_console/include -I/Users/usuario/esp/idf/esp-idf/components/esp_http_server/include -I/Users/usuario/esp/idf/esp-idf/components/http_parser -I/Users/usuario/esp/idf/esp-idf/components/driver/deprecated -I/Users/usuario/esp/idf/esp-idf/components/driver/i2c/include -I/Users/usuario/esp/idf/esp-idf/components/driver/touch_sensor/include -I/Users/usuario/esp/idf/esp-idf/components/driver/twai/include -I/Users/usuario/esp/idf/esp-idf/components/driver/touch_sensor/esp32/include -I/Users/usuario/esp/idf/esp-idf/components/esp_pm/include -I/Users/usuario/esp/idf/esp-idf/components/esp_ringbuf/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_gpio/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_pcnt/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_gptimer/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_spi/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_mcpwm/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_ana_cmpr/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_i2s/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_sdmmc/include -I/Users/usuario/esp/idf/esp-idf/components/sdmmc/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_sdspi/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_sdio/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_dac/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_rmt/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_tsens/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_sdm/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_i2c/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_uart/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_ledc/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_parlio/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_usb_serial_jtag/include

C_FLAGS = -mlongcalls -Wno-frame-address  -fno-builtin-memcpy -fno-builtin-memset -fno-builtin-bzero -fno-builtin-stpcpy -fno-builtin-strncpy -fdiagnostics-color=always -ffunction-sections -fdata-sections -Wall -Werror=all -Wno-error=unused-function -Wno-error=unused-variable -Wno-error=unused-but-set-variable -Wno-error=deprecated-declarations -Wextra -Wno-unused-parameter -Wno-sign-compare -Wno-enum-conversion -gdwarf-4 -ggdb -mfix-esp32-psram-cache-issue -mfix-esp32-psram-cache-strategy=memw -O2 -fmacro-prefix-map=/Users/usuario/esp/projects_tf/person_detection=. -fmacro-prefix-map=/Users/usuario/esp/idf/esp-idf=/IDF -fstrict-volatile-bitfields -fno-jump-tables -fno-tree-switch-conversion -std=gnu17 -Wno-old-style-declaration

