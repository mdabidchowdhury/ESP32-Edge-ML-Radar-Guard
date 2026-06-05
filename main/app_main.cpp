#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"

#include "nvs_flash.h"
#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"

#include "lwip/inet.h"
#include "lwip/netdb.h"
#include "lwip/sockets.h"
#include "ping/ping_sock.h"

#include "protocol_examples_common.h"
#include "esp_csi_gain_ctrl.h"

// --- TensorFlow Lite Includes ---
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"

#define CONFIG_SEND_FREQUENCY      100
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
#define CSI_FORCE_LLTF             0
#endif
#define CONFIG_FORCE_GAIN          0

#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
#define CONFIG_GAIN_CONTROL        1
#endif

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define ESP_IF_WIFI_STA ESP_MAC_WIFI_STA
#endif

static const char *TAG = "csi_recv_router";

// Reference the model you generated
extern const unsigned char g_model[];
extern const int g_model_len;

// TFLite Global Variables
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 10 * 1024; // 10KB memory arena
uint8_t tensor_arena[kTensorArenaSize];

// Sliding window buffer for the 10 timesteps of CSI data
float csi_buffer[10][64] = {0};
int buffer_index = 0;


static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || interpreter == nullptr) return;

    int8_t *raw_data = info->buf;
    
    // 1. Calculate the amplitude of the 64 subcarriers and push to our sliding window
    for (int i = 0; i < 64; i++) {
        int8_t i_val = raw_data[i * 2];
        int8_t q_val = raw_data[i * 2 + 1];
        csi_buffer[buffer_index][i] = sqrt((i_val * i_val) + (q_val * q_val));
    }
    
    buffer_index = (buffer_index + 1) % 10; // Move window forward

    // 2. Load the 10-step history into the TFLite Input Tensor
    int tensor_idx = 0;
    for (int step = 0; step < 10; step++) {
        int actual_idx = (buffer_index + step) % 10;
        for (int sub = 0; sub < 64; sub++) {
            // Convert float to INT8 for the quantized model
            input->data.int8[tensor_idx++] = (int8_t)(csi_buffer[actual_idx][sub] - 128); 
        }
    }

    // 3. Run AI Inference!
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE("AI", "Inference failed");
        return;
    }

    // 4. Read the Agent's Action
    // Action 0 = Sleep/Clear, Action 1 = Alarm/Intruder
    int8_t action_0_score = output->data.int8[0];
    int8_t action_1_score = output->data.int8[1];

    if (action_1_score > action_0_score) {
        ESP_LOGW("RADAR", "🚨 INTRUDER DETECTED! Wi-Fi waves disrupted! 🚨");
    } else {
        ESP_LOGI("RADAR", "Room is clear. Conserving power.");
    }
}

static void wifi_csi_init()
{
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
    wifi_csi_config_t csi_config = {
        .enable                   = true,
        .acquire_csi_legacy       = true,
        .acquire_csi_force_lltf   = CSI_FORCE_LLTF,
        .acquire_csi_ht20         = true,
        .acquire_csi_ht40         = true,
        .acquire_csi_vht          = false,
        .acquire_csi_su           = false,
        .acquire_csi_mu           = false,
        .acquire_csi_dcm          = false,
        .acquire_csi_beamformed   = false,
        .acquire_csi_he_stbc_mode = 2,
        .val_scale_cfg            = 0,
        .dump_ack_en              = false,
        .reserved                 = false
    };
#elif CONFIG_IDF_TARGET_ESP32C6
    wifi_csi_config_t csi_config = {
        .enable                 = true,
        .acquire_csi_legacy     = true,
        .acquire_csi_ht20       = true,
        .acquire_csi_ht40       = true,
        .acquire_csi_su         = false,
        .acquire_csi_mu         = false,
        .acquire_csi_dcm        = false,
        .acquire_csi_beamformed = false,
        .acquire_csi_he_stbc    = 2,
        .val_scale_cfg          = false,
        .dump_ack_en            = false,
        .reserved               = false
    };
#else
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = false,
        .stbc_htltf2_en    = false,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = true,
        .shift             = true,
    };
#endif
    static wifi_ap_record_t s_ap_info = {0};
    ESP_ERROR_CHECK(esp_wifi_sta_get_ap_info(&s_ap_info));
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, s_ap_info.bssid));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

static esp_err_t wifi_ping_router_start()
{
    static esp_ping_handle_t ping_handle = NULL;

    esp_ping_config_t ping_config = ESP_PING_DEFAULT_CONFIG();
    ping_config.count             = 0;
    ping_config.interval_ms       = 1000 / CONFIG_SEND_FREQUENCY;
    ping_config.task_stack_size   = 3072;
    ping_config.data_size         = 1;

    esp_netif_ip_info_t local_ip;
    esp_netif_get_ip_info(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"), &local_ip);
    ESP_LOGI(TAG, "got ip:" IPSTR ", gw: " IPSTR, IP2STR(&local_ip.ip), IP2STR(&local_ip.gw));
    ping_config.target_addr.u_addr.ip4.addr = ip4_addr_get_u32(&local_ip.gw);
    ping_config.target_addr.type = ESP_IPADDR_TYPE_V4;

    esp_ping_callbacks_t cbs = { 0 };
    esp_ping_new_session(&ping_config, &cbs, &ping_handle);
    esp_ping_start(ping_handle);

    return ESP_OK;
}

// CRITICAL: Must be extern "C" for ESP-IDF to recognize the entry point in a .cpp file
extern "C" void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    ESP_ERROR_CHECK(example_connect());

    // --- Boot the AI Brain ---
    ESP_LOGI("AI", "Initializing TensorFlow Lite Micro...");
    tflite::InitializeTarget();
    model = tflite::GetModel(g_model);

    // Load the mathematical operations our model needs
    static tflite::MicroMutableOpResolver<3> micro_op_resolver;
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddTanh();

    // Build the interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
    
    input = interpreter->input(0);
    output = interpreter->output(0);
    ESP_LOGI("AI", "Brain successfully booted!");
    // -------------------------

    wifi_csi_init();
    wifi_ping_router_start();
}