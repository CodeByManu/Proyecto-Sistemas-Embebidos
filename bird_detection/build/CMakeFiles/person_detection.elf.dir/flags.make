# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# compile C with /Users/usuario/esp/idf-tools/tools/xtensa-esp-elf/esp-13.2.0_20230928/xtensa-esp-elf/bin/xtensa-esp32-elf-gcc
C_DEFINES = -DMBEDTLS_CONFIG_FILE=\"mbedtls/esp_config.h\" -DSOC_MMU_PAGE_SIZE=CONFIG_MMU_PAGE_SIZE -DSOC_XTAL_FREQ_MHZ=CONFIG_XTAL_FREQ -DUNITY_INCLUDE_CONFIG_H

C_INCLUDES = -I/Users/usuario/esp/idf/esp-idf/components/xtensa/esp32/include -I/Users/usuario/esp/idf/esp-idf/components/xtensa/include -I/Users/usuario/esp/idf/esp-idf/components/xtensa/deprecated_include -I/Users/usuario/esp/projects_tf/person_detection/build/config -I/Users/usuario/esp/idf/esp-idf/components/newlib/platform_include -I/Users/usuario/esp/idf/esp-idf/components/freertos/config/include -I/Users/usuario/esp/idf/esp-idf/components/freertos/config/include/freertos -I/Users/usuario/esp/idf/esp-idf/components/freertos/config/xtensa/include -I/Users/usuario/esp/idf/esp-idf/components/freertos/FreeRTOS-Kernel/include -I/Users/usuario/esp/idf/esp-idf/components/freertos/FreeRTOS-Kernel/portable/xtensa/include -I/Users/usuario/esp/idf/esp-idf/components/freertos/FreeRTOS-Kernel/portable/xtensa/include/freertos -I/Users/usuario/esp/idf/esp-idf/components/freertos/esp_additions/include -I/Users/usuario/esp/idf/esp-idf/components/esp_hw_support/include -I/Users/usuario/esp/idf/esp-idf/components/esp_hw_support/include/soc -I/Users/usuario/esp/idf/esp-idf/components/esp_hw_support/include/soc/esp32 -I/Users/usuario/esp/idf/esp-idf/components/esp_hw_support/dma/include -I/Users/usuario/esp/idf/esp-idf/components/esp_hw_support/port/esp32/. -I/Users/usuario/esp/idf/esp-idf/components/esp_hw_support/port/esp32/include -I/Users/usuario/esp/idf/esp-idf/components/heap/include -I/Users/usuario/esp/idf/esp-idf/components/log/include -I/Users/usuario/esp/idf/esp-idf/components/soc/include -I/Users/usuario/esp/idf/esp-idf/components/soc/esp32 -I/Users/usuario/esp/idf/esp-idf/components/soc/esp32/include -I/Users/usuario/esp/idf/esp-idf/components/hal/platform_port/include -I/Users/usuario/esp/idf/esp-idf/components/hal/esp32/include -I/Users/usuario/esp/idf/esp-idf/components/hal/include -I/Users/usuario/esp/idf/esp-idf/components/esp_rom/include -I/Users/usuario/esp/idf/esp-idf/components/esp_rom/include/esp32 -I/Users/usuario/esp/idf/esp-idf/components/esp_rom/esp32 -I/Users/usuario/esp/idf/esp-idf/components/esp_common/include -I/Users/usuario/esp/idf/esp-idf/components/esp_system/include -I/Users/usuario/esp/idf/esp-idf/components/esp_system/port/soc -I/Users/usuario/esp/idf/esp-idf/components/esp_system/port/include/private -I/Users/usuario/esp/idf/esp-idf/components/esp_timer/include -I/Users/usuario/esp/idf/esp-idf/components/lwip/include -I/Users/usuario/esp/idf/esp-idf/components/lwip/include/apps -I/Users/usuario/esp/idf/esp-idf/components/lwip/include/apps/sntp -I/Users/usuario/esp/idf/esp-idf/components/lwip/lwip/src/include -I/Users/usuario/esp/idf/esp-idf/components/lwip/port/include -I/Users/usuario/esp/idf/esp-idf/components/lwip/port/freertos/include -I/Users/usuario/esp/idf/esp-idf/components/lwip/port/esp32xx/include -I/Users/usuario/esp/idf/esp-idf/components/lwip/port/esp32xx/include/arch -I/Users/usuario/esp/idf/esp-idf/components/lwip/port/esp32xx/include/sys -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_gpio/include -I/Users/usuario/esp/idf/esp-idf/components/esp_pm/include -I/Users/usuario/esp/idf/esp-idf/components/mbedtls/port/include -I/Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/include -I/Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/library -I/Users/usuario/esp/idf/esp-idf/components/mbedtls/esp_crt_bundle/include -I/Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/3rdparty/everest/include -I/Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/3rdparty/p256-m -I/Users/usuario/esp/idf/esp-idf/components/mbedtls/mbedtls/3rdparty/p256-m/p256-m -I/Users/usuario/esp/idf/esp-idf/components/esp_app_format/include -I/Users/usuario/esp/idf/esp-idf/components/esp_bootloader_format/include -I/Users/usuario/esp/idf/esp-idf/components/app_update/include -I/Users/usuario/esp/idf/esp-idf/components/bootloader_support/include -I/Users/usuario/esp/idf/esp-idf/components/bootloader_support/bootloader_flash/include -I/Users/usuario/esp/idf/esp-idf/components/esp_partition/include -I/Users/usuario/esp/idf/esp-idf/components/efuse/include -I/Users/usuario/esp/idf/esp-idf/components/efuse/esp32/include -I/Users/usuario/esp/idf/esp-idf/components/esp_mm/include -I/Users/usuario/esp/idf/esp-idf/components/spi_flash/include -I/Users/usuario/esp/idf/esp-idf/components/pthread/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_gptimer/include -I/Users/usuario/esp/idf/esp-idf/components/esp_ringbuf/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_uart/include -I/Users/usuario/esp/idf/esp-idf/components/vfs/include -I/Users/usuario/esp/idf/esp-idf/components/app_trace/include -I/Users/usuario/esp/idf/esp-idf/components/esp_event/include -I/Users/usuario/esp/idf/esp-idf/components/nvs_flash/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_pcnt/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_spi/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_mcpwm/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_ana_cmpr/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_i2s/include -I/Users/usuario/esp/idf/esp-idf/components/sdmmc/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_sdmmc/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_sdspi/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_sdio/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_dac/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_rmt/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_tsens/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_sdm/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_i2c/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_ledc/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_parlio/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_usb_serial_jtag/include -I/Users/usuario/esp/idf/esp-idf/components/driver/deprecated -I/Users/usuario/esp/idf/esp-idf/components/driver/i2c/include -I/Users/usuario/esp/idf/esp-idf/components/driver/touch_sensor/include -I/Users/usuario/esp/idf/esp-idf/components/driver/twai/include -I/Users/usuario/esp/idf/esp-idf/components/driver/touch_sensor/esp32/include -I/Users/usuario/esp/idf/esp-idf/components/esp_phy/include -I/Users/usuario/esp/idf/esp-idf/components/esp_phy/esp32/include -I/Users/usuario/esp/idf/esp-idf/components/esp_vfs_console/include -I/Users/usuario/esp/idf/esp-idf/components/esp_netif/include -I/Users/usuario/esp/idf/esp-idf/components/wpa_supplicant/include -I/Users/usuario/esp/idf/esp-idf/components/wpa_supplicant/port/include -I/Users/usuario/esp/idf/esp-idf/components/wpa_supplicant/esp_supplicant/include -I/Users/usuario/esp/idf/esp-idf/components/esp_coex/include -I/Users/usuario/esp/idf/esp-idf/components/esp_wifi/include -I/Users/usuario/esp/idf/esp-idf/components/esp_wifi/wifi_apps/include -I/Users/usuario/esp/idf/esp-idf/components/esp_wifi/include/local -I/Users/usuario/esp/idf/esp-idf/components/unity/include -I/Users/usuario/esp/idf/esp-idf/components/unity/unity/src -I/Users/usuario/esp/idf/esp-idf/components/cmock/CMock/src -I/Users/usuario/esp/idf/esp-idf/components/console -I/Users/usuario/esp/idf/esp-idf/components/http_parser -I/Users/usuario/esp/idf/esp-idf/components/esp-tls -I/Users/usuario/esp/idf/esp-idf/components/esp-tls/esp-tls-crypto -I/Users/usuario/esp/idf/esp-idf/components/esp_adc/include -I/Users/usuario/esp/idf/esp-idf/components/esp_adc/interface -I/Users/usuario/esp/idf/esp-idf/components/esp_adc/esp32/include -I/Users/usuario/esp/idf/esp-idf/components/esp_adc/deprecated/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_cam/include -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_cam/interface -I/Users/usuario/esp/idf/esp-idf/components/esp_driver_jpeg/include -I/Users/usuario/esp/idf/esp-idf/components/esp_eth/include -I/Users/usuario/esp/idf/esp-idf/components/esp_gdbstub/include -I/Users/usuario/esp/idf/esp-idf/components/esp_hid/include -I/Users/usuario/esp/idf/esp-idf/components/tcp_transport/include -I/Users/usuario/esp/idf/esp-idf/components/esp_http_client/include -I/Users/usuario/esp/idf/esp-idf/components/esp_http_server/include -I/Users/usuario/esp/idf/esp-idf/components/esp_https_ota/include -I/Users/usuario/esp/idf/esp-idf/components/esp_https_server/include -I/Users/usuario/esp/idf/esp-idf/components/esp_psram/include -I/Users/usuario/esp/idf/esp-idf/components/esp_lcd/include -I/Users/usuario/esp/idf/esp-idf/components/esp_lcd/interface -I/Users/usuario/esp/idf/esp-idf/components/protobuf-c/protobuf-c -I/Users/usuario/esp/idf/esp-idf/components/protocomm/include/common -I/Users/usuario/esp/idf/esp-idf/components/protocomm/include/security -I/Users/usuario/esp/idf/esp-idf/components/protocomm/include/transports -I/Users/usuario/esp/idf/esp-idf/components/protocomm/include/crypto/srp6a -I/Users/usuario/esp/idf/esp-idf/components/protocomm/proto-c -I/Users/usuario/esp/idf/esp-idf/components/esp_local_ctrl/include -I/Users/usuario/esp/idf/esp-idf/components/espcoredump/include -I/Users/usuario/esp/idf/esp-idf/components/espcoredump/include/port/xtensa -I/Users/usuario/esp/idf/esp-idf/components/wear_levelling/include -I/Users/usuario/esp/idf/esp-idf/components/fatfs/diskio -I/Users/usuario/esp/idf/esp-idf/components/fatfs/src -I/Users/usuario/esp/idf/esp-idf/components/fatfs/vfs -I/Users/usuario/esp/idf/esp-idf/components/idf_test/include -I/Users/usuario/esp/idf/esp-idf/components/idf_test/include/esp32 -I/Users/usuario/esp/idf/esp-idf/components/ieee802154/include -I/Users/usuario/esp/idf/esp-idf/components/json/cJSON -I/Users/usuario/esp/idf/esp-idf/components/mqtt/esp-mqtt/include -I/Users/usuario/esp/idf/esp-idf/components/nvs_sec_provider/include -I/Users/usuario/esp/idf/esp-idf/components/perfmon/include -I/Users/usuario/esp/idf/esp-idf/components/spiffs/include -I/Users/usuario/esp/idf/esp-idf/components/wifi_provisioning/include -I/Users/usuario/esp/projects_tf/person_detection/managed_components/espressif__esp-nn/include -I/Users/usuario/esp/projects_tf/person_detection/managed_components/espressif__esp-nn/src/common -I/Users/usuario/esp/projects_tf/person_detection/managed_components/espressif__esp-tflite-micro -I/Users/usuario/esp/projects_tf/person_detection/managed_components/espressif__esp-tflite-micro/third_party/gemmlowp -I/Users/usuario/esp/projects_tf/person_detection/managed_components/espressif__esp-tflite-micro/third_party/flatbuffers/include -I/Users/usuario/esp/projects_tf/person_detection/managed_components/espressif__esp-tflite-micro/third_party/ruy -I/Users/usuario/esp/projects_tf/person_detection/managed_components/espressif__esp-tflite-micro/third_party/kissfft -I/Users/usuario/esp/projects_tf/person_detection/managed_components/espressif__esp-tflite-micro/signal/micro/kernels -I/Users/usuario/esp/projects_tf/person_detection/managed_components/espressif__esp-tflite-micro/signal/src -I/Users/usuario/esp/projects_tf/person_detection/managed_components/espressif__esp-tflite-micro/signal/src/kiss_fft_wrappers -I/Users/usuario/esp/projects_tf/person_detection/managed_components/espressif__esp32-camera/driver/include -I/Users/usuario/esp/projects_tf/person_detection/managed_components/espressif__esp32-camera/conversions/include

C_FLAGS = -mlongcalls -Wno-frame-address  -fno-builtin-memcpy -fno-builtin-memset -fno-builtin-bzero -fno-builtin-stpcpy -fno-builtin-strncpy -fdiagnostics-color=always -mfix-esp32-psram-cache-issue -mfix-esp32-psram-cache-strategy=memw -DTF_LITE_STATIC_MEMORY

