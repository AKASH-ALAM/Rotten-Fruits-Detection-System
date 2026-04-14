#include "board_config.h"
#include "esp_camera.h"
#include "soc/rtc_cntl_reg.h"
#include "soc/soc.h"
#include <HTTPClient.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>

// --- YOUR CONFIGURATION ---
char *ssid = "WiFi_Name";
char *password = "Your_WiFi_Pass";

String serverName = "https://user_name-project_name-api.hf.space/predict";
String checkUrl = "https://user_name-project_name-api.hf.space/check";

unsigned long lastCaptureTime = 0;
const unsigned long timerInterval = 3600000; // 1 Hour

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  Serial.println("\nBooting up System...");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // AI OPTIMIZATION (Matches YOLOv8 natively for fast uploads)
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 6;
  config.fb_count = 1;

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 1);
}

void loop() {
  // 1. WI-FI AUTO-RECONNECT
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi Disconnected. Reconnecting...");
    WiFi.disconnect();
    WiFi.reconnect();
    delay(5000);
    return;
  }

  bool takePhotoNow = false;
  unsigned long currentMillis = millis();

  // 2. CHECK TIMER
  if ((currentMillis - lastCaptureTime >= timerInterval) ||
      lastCaptureTime == 0) {
    takePhotoNow = true;
  }
  // 3. CHECK HUGGING FACE FOR TELEGRAM COMMAND
  else {
    WiFiClientSecure *client = new WiFiClientSecure;
    client->setInsecure();
    HTTPClient httpCheck;
    httpCheck.begin(*client, checkUrl);

    int checkCode = httpCheck.GET();
    if (checkCode == 200) {
      String command = httpCheck.getString();
      command.trim(); // THE FIX: Strips invisible newlines

      if (command == "YES") {
        Serial.println("📱 Command Received!");
        takePhotoNow = true;
      }
    } else {
      Serial.printf("Server Check Failed. Code: %d\n", checkCode);
    }
    httpCheck.end();
    delete client;
  }

  // 4. CAPTURE AND UPLOAD
  if (takePhotoNow) {
    Serial.println("Waking camera...");

    // Clear stale frame
    camera_fb_t *dummy_fb = esp_camera_fb_get();
    if (dummy_fb) {
      esp_camera_fb_return(dummy_fb);
    }
    delay(200);

    Serial.println("Capturing...");
    camera_fb_t *fb = esp_camera_fb_get();

    if (fb) {
      WiFiClientSecure *uploadClient = new WiFiClientSecure;
      uploadClient->setInsecure();

      String boundary = "Esp32MultipartBoundary";
      String bodyStart = "--" + boundary + "\r\n";
      bodyStart += "Content-Disposition: form-data; name=\"file\"; "
                   "filename=\"photo.jpg\"\r\n";
      bodyStart += "Content-Type: image/jpeg\r\n\r\n";
      String bodyEnd = "\r\n--" + boundary + "--\r\n";

      size_t totalLength = bodyStart.length() + fb->len + bodyEnd.length();
      uint8_t *full_payload = (uint8_t *)ps_malloc(totalLength);

      if (full_payload != NULL) {
        memcpy(full_payload, bodyStart.c_str(), bodyStart.length());
        memcpy(full_payload + bodyStart.length(), fb->buf, fb->len);
        memcpy(full_payload + bodyStart.length() + fb->len, bodyEnd.c_str(),
               bodyEnd.length());

        HTTPClient httpUpload;
        httpUpload.begin(*uploadClient, serverName);
        httpUpload.addHeader("Content-Type",
                             "multipart/form-data; boundary=" + boundary);

        httpUpload.setTimeout(20000); // Wait 20s for AI to process

        int responseCode = httpUpload.POST(full_payload, totalLength);
        if (responseCode > 0) {
          Serial.println("AI Success!");
        } else {
          Serial.printf("Upload failed. Code: %d\n", responseCode);
        }
        httpUpload.end();
        free(full_payload);
      }
      esp_camera_fb_return(fb);
      delete uploadClient;
    }
    lastCaptureTime = millis();
  }

  // THE SWEET SPOT: Wait 2 seconds. Fast enough to feel responsive,
  // slow enough to prevent Hugging Face from blocking your IP.
  delay(2000);
}