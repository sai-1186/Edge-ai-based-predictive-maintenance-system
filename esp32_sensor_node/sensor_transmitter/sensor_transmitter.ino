/*
 * ==============================================================================
 * Edge AI Predictive Maintenance - Sensor Node & Edge AI Transmitter
 * ==============================================================================
 * 
 * Hardware Architecture:
 * - Microcontroller: ESP32 Dev Module (WROOM-32)
 * - Sensors:
 *     - ACS712 (Current Sensor): Pin 34 (ADC1_CH6, input only)
 *     - MPU6050 (3-Axis Accelerometer/Vibration): I2C SDA=21, SCL=22
 *     - DHT11 (Temperature & Humidity): Pin 4
 * - Display:
 *     - ST7735 1.8" TFT (160x128 SPI): CS=5, DC=16, RST=17, MOSI=23, SCK=18
 * - Edge AI Model:
 *     - TensorFlow Lite Micro (TensorFlowLite_ESP32)
 *     - predmodel.h (Embedded TFLite ANN model, 6 inputs, 3 classes)
 *     - predlabel.h (Labels: "Critical", "Normal", "Warning")
 *     - StandardScaler fitted parameters from training dataset
 * 
 * Flow:
 *   Sensors -> Normalization -> On-Chip TFLite Inference -> TFT Display + Wi-Fi JSON
 * ==============================================================================
 */

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <WiFi.h>
#include <HTTPClient.h>

// Graphics and Sensors
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>

// TensorFlow Lite Micro for ESP32
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Embedded Model & Labels
#include "predmodel.h"
#include "predlabel.h"

// ==============================================================================
// 1. PIN CONFIGURATION & HARDWARE CONSTANTS
// ==============================================================================
constexpr uint8_t TFT_CS            = 5;
constexpr uint8_t TFT_DC            = 16;
constexpr uint8_t TFT_RST           = 17;
constexpr uint8_t DHT_PIN           = 4;
constexpr uint8_t CURRENT_SENSOR_PIN = 34; // ADC1 channel (usable with Wi-Fi active)
constexpr uint8_t I2C_SDA           = 21;
constexpr uint8_t I2C_SCL           = 22;

constexpr uint8_t DHT_TYPE          = DHT11;
constexpr float ADC_MAX             = 4095.0f;
constexpr float VREF                = 3.3f;
constexpr float ACS712_SENSITIVITY  = 0.100f;  // 20A module: 100 mV/A (0.100 V/A)
constexpr float STANDARD_GRAVITY    = 9.80665f; // m/s² per 1g

// Timing intervals (milliseconds)
constexpr unsigned long ACS_INTERVAL     = 400;
constexpr unsigned long MPU_INTERVAL     = 100;
constexpr unsigned long DHT_INTERVAL     = 2000;
constexpr unsigned long INFERENCE_INTERVAL = 1000;
constexpr unsigned long DISPLAY_INTERVAL = 500;
constexpr unsigned long SEND_INTERVAL    = 2000;
constexpr unsigned long WIFI_RETRY_INTERVAL = 10000;

// ==============================================================================
// 2. NETWORK CONFIGURATION
// ==============================================================================
const char* ssid      = "Sai's net";
const char* password  = "sai@1186";
const char* serverUrl = "http://192.168.32.237:8080/api/data";
const char* machineId = "MOTOR-01";

// ==============================================================================
// 3. MODEL PREPROCESSING: STANDARD SCALER CONSTANTS
// Form: scaled = (raw_value - mean) / scale
// Feature Order: [0] Temperature_C, [1] Humidity_%, [2] Vib_X, [3] Vib_Y, [4] Vib_Z, [5] Current_A
// ==============================================================================
constexpr float FEATURE_MEANS[6] = {
    46.96244444f, // Temperature_C
    58.23511111f, // Humidity_%
    0.00986667f,  // Vibration_X_g
    0.01003556f,  // Vibration_Y_g
    0.91876222f,  // Vibration_Z_g
    8.10116000f   // Current_A
};

constexpr float FEATURE_SCALES[6] = {
    17.54411724f, // Temperature_C
    12.42588921f, // Humidity_%
    0.75860524f,  // Vibration_X_g
    0.75716575f,  // Vibration_Y_g
    0.24906349f,  // Vibration_Z_g
    5.63859483f   // Current_A
};

// ==============================================================================
// 4. GLOBAL OBJECTS & TFLITE GLOBALS
// ==============================================================================
Adafruit_ST7735 tft(TFT_CS, TFT_DC, TFT_RST);
Adafruit_MPU6050 mpu;
DHT dht(DHT_PIN, DHT_TYPE);

// TensorFlow Lite Micro runtime
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// 8 KB tensor arena is plenty for this 6-input, 3-output neural network
constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
} // namespace

// System state variables
bool aiAvailable     = false;
bool mpuAvailable    = false;
int  acsZeroRaw      = 2048;

float lastTempC     = 25.0f;
float lastHum       = 50.0f;
float lastCurrentA  = 0.0f;
float lastAccelX    = 0.0f;
float lastAccelY    = 0.0f;
float lastAccelZ    = 0.98f;

int   predictedClassIndex = 1; // Default to Normal (index 1)
char  machineHealthStr[24] = "Normal";

// Timing trackers
unsigned long lastACSMillis         = 0;
unsigned long lastMPUMillis         = 0;
unsigned long lastDHTMillis         = 0;
unsigned long lastInferenceMillis   = 0;
unsigned long lastDisplayMillis     = 0;
unsigned long lastSendMillis        = 0;
unsigned long lastWiFiAttemptMillis = 0;

// UI Cache to prevent unnecessary redraw flicker
float prevTemp     = -999.0f;
float prevHum      = -999.0f;
float prevCurrent  = -999.0f;
float prevX        = -999.0f;
float prevY        = -999.0f;
float prevZ        = -999.0f;
int   prevHealthIdx = -1;
bool  prevWiFiStatus = false;

// 16-bit 565 Custom Colors for Industrial Dashboard
#define COLOR_BG         0x0882  // Deep dark navy
#define COLOR_CARD_BG    0x1905  // Slate dark blue
#define COLOR_BAR_BG     0x10A2  // Header/footer bar
#define COLOR_BORDER     0x2967  // Slate border
#define COLOR_NORMAL     0x07E0  // Crisp green
#define COLOR_WARNING    0xFD20  // Industrial orange/amber
#define COLOR_CRITICAL   0xF800  // Red alert
#define COLOR_CYAN_ACC   0x07FF  // Cyan
#define COLOR_WHITE_TEXT 0xFFFF
#define COLOR_MUTED_TEXT 0x9CD3

// ==============================================================================
// 5. FUNCTION DECLARATIONS
// ==============================================================================
void setupTensorFlowLite();
void calibrateACS712(uint16_t samples);
float readACS712Current(uint16_t samples);
void readSensors(unsigned long now);
void runAiInference();
void handleWiFiReconnection(unsigned long now);
void sendSensorData();
void drawStaticLayout();
void drawDynamicValues();

// Lightweight Primitive Icon Helpers (zero image assets, highly memory-efficient)
void drawThermometerIcon(int16_t x, int16_t y, uint16_t color);
void drawDropletIcon(int16_t x, int16_t y, uint16_t color);
void drawCurrentIcon(int16_t x, int16_t y, uint16_t color);
void drawVibrationIcon(int16_t x, int16_t y, uint16_t color);
void drawWiFiIcon(int16_t x, int16_t y, bool connected);

// ==============================================================================
// 6. SETUP
// ==============================================================================
void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n--- [ESP32 EDGE AI PREDICTIVE MAINTENANCE] ---");

  // 1. Initialize Display
  tft.initR(INITR_BLACKTAB);
  tft.setRotation(1); // 160 x 128 landscape
  tft.fillScreen(COLOR_BG);
  tft.setTextWrap(false);

  tft.setTextColor(COLOR_WHITE_TEXT);
  tft.setTextSize(1);
  tft.setCursor(10, 20);
  tft.print("Booting Edge AI...");

  // 2. Initialize I2C and Sensors
  Wire.begin(I2C_SDA, I2C_SCL);
  dht.begin();
  analogSetWidth(12);
  analogSetPinAttenuation(CURRENT_SENSOR_PIN, ADC_11db);

  mpuAvailable = mpu.begin();
  if (mpuAvailable) {
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    Serial.println("MPU6050 initialized successfully.");
  } else {
    Serial.println("WARNING: MPU6050 not detected. Using fallback baseline.");
  }

  // 3. Calibrate Current Sensor Baseline
  tft.setCursor(10, 40);
  tft.print("Calibrating ACS712...");
  calibrateACS712(400);

  // 4. Initialize TensorFlow Lite Micro ON-CHIP
  tft.setCursor(10, 60);
  tft.print("Loading TFLite ANN...");
  setupTensorFlowLite();

  // 5. Initial Non-blocking Wi-Fi Attempt
  tft.setCursor(10, 80);
  tft.print("Connecting Wi-Fi...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  unsigned long wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - wifiStart < 3000) {
    delay(100);
  }
  lastWiFiAttemptMillis = millis();

  // 6. Draw Dashboard Static UI Frame
  drawStaticLayout();
  lastDHTMillis = millis() - DHT_INTERVAL;
}

// ==============================================================================
// 7. MAIN LOOP
// Non-blocking architecture: AI, Sensors, and TFT run uninterrupted at all times.
// ==============================================================================
void loop() {
  const unsigned long now = millis();

  // 1. Sensor Acquisition
  readSensors(now);

  // 2. TensorFlow Lite Machine Health Inference on ESP32
  if (now - lastInferenceMillis >= INFERENCE_INTERVAL) {
    runAiInference();
    lastInferenceMillis = now;
  }

  // 3. Update Innovative Industrial TFT UI
  if (now - lastDisplayMillis >= DISPLAY_INTERVAL) {
    drawDynamicValues();
    lastDisplayMillis = now;
  }

  // 4. Non-blocking Wi-Fi Reconnection Handler
  handleWiFiReconnection(now);

  // 5. Transmit JSON ONLY when Wi-Fi is Connected
  if (WiFi.status() == WL_CONNECTED && (now - lastSendMillis >= SEND_INTERVAL)) {
    sendSensorData();
    lastSendMillis = now;
  }
}

// ==============================================================================
// 8. TENSORFLOW LITE MICRO INITIALIZATION
// ==============================================================================
void setupTensorFlowLite() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model from predmodel.h
  model = tflite::GetModel(predmodel);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("TFLite Error: Model schema %d != Runtime %d\n", model->version(), TFLITE_SCHEMA_VERSION);
    aiAvailable = false;
    return;
  }

  // Pull in all necessary operations (MatMul, Relu, BiasAdd, Softmax)
  static tflite::AllOpsResolver resolver;

  // Build micro interpreter with dedicated tensor arena
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from arena for model input/output tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("TFLite Error: AllocateTensors() failed.");
    aiAvailable = false;
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  if (input == nullptr || output == nullptr) {
    Serial.println("TFLite Error: Failed to obtain input/output tensors.");
    aiAvailable = false;
    return;
  }

  aiAvailable = true;
  Serial.println("TensorFlow Lite Micro initialized successfully on ESP32!");
  Serial.printf("Input Tensor: %d dims, Output Tensor: %d dims (%d classes)\n",
                input->dims->data[1], output->dims->data[1], NUM_CLASSES);
}

// ==============================================================================
// 9. ON-DEVICE AI INFERENCE & PREPROCESSING
// ==============================================================================
void runAiInference() {
  if (!aiAvailable || interpreter == nullptr || input == nullptr || output == nullptr) {
    snprintf(machineHealthStr, sizeof(machineHealthStr), "AI: ERROR");
    predictedClassIndex = -1;
    return;
  }

  // Prepare 6 raw features
  // Ensure valid numeric fallbacks if a sensor reading had transient timeout
  const float t = isnan(lastTempC) ? 25.0f : lastTempC;
  const float h = isnan(lastHum) ? 50.0f : lastHum;
  const float vx = lastAccelX;
  const float vy = lastAccelY;
  const float vz = lastAccelZ;
  const float cur = lastCurrentA;

  const float rawFeatures[6] = { t, h, vx, vy, vz, cur };

  // Apply EXACT StandardScaler normalization: scaled = (raw - mean) / scale
  for (int i = 0; i < 6; ++i) {
    input->data.f[i] = (rawFeatures[i] - FEATURE_MEANS[i]) / FEATURE_SCALES[i];
  }

  // Invoke TensorFlow Lite Model on ESP32
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("TFLite Error: Invoke() failed.");
    snprintf(machineHealthStr, sizeof(machineHealthStr), "AI: ERROR");
    return;
  }

  // Class determination via argmax over output probabilities
  int bestIndex = 0;
  float maxScore = output->data.f[0];
  for (int i = 1; i < NUM_CLASSES; ++i) {
    if (output->data.f[i] > maxScore) {
      maxScore = output->data.f[i];
      bestIndex = i;
    }
  }

  predictedClassIndex = bestIndex;

  // Map to human-readable label from predlabel.h (0: Critical, 1: Normal, 2: Warning)
  if (bestIndex >= 0 && bestIndex < NUM_CLASSES) {
    snprintf(machineHealthStr, sizeof(machineHealthStr), "%s", labels[bestIndex]);
  } else {
    snprintf(machineHealthStr, sizeof(machineHealthStr), "Normal");
  }
}

// ==============================================================================
// 10. SENSOR ACQUISITION & CALIBRATION
// ==============================================================================
void calibrateACS712(uint16_t samples) {
  uint32_t sum = 0;
  for (uint16_t i = 0; i < samples; ++i) {
    sum += analogRead(CURRENT_SENSOR_PIN);
    delay(1);
  }
  acsZeroRaw = sum / samples;
  Serial.printf("ACS712 Baseline Zero Raw: %d (%.3f V)\n", acsZeroRaw, acsZeroRaw * VREF / ADC_MAX);
}

float readACS712Current(uint16_t samples) {
  uint32_t sum = 0;
  for (uint16_t i = 0; i < samples; ++i) {
    sum += analogRead(CURRENT_SENSOR_PIN);
  }
  const float average = static_cast<float>(sum) / samples;
  const float voltageDelta = (average - acsZeroRaw) * VREF / ADC_MAX;
  float current = abs(voltageDelta / ACS712_SENSITIVITY);
  if (current < 0.15f) current = 0.0f; // Small noise floor filter
  return current;
}

void readSensors(unsigned long now) {
  // ACS712 Current Read
  if (now - lastACSMillis >= ACS_INTERVAL) {
    lastCurrentA = readACS712Current(50);
    lastACSMillis = now;
  }

  // MPU6050 Vibration / Acceleration Read
  if (mpuAvailable && (now - lastMPUMillis >= MPU_INTERVAL)) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    // Convert m/s² to standard g-force
    lastAccelX = a.acceleration.x / STANDARD_GRAVITY;
    lastAccelY = a.acceleration.y / STANDARD_GRAVITY;
    lastAccelZ = a.acceleration.z / STANDARD_GRAVITY;
    lastMPUMillis = now;
  }

  // DHT11 Temperature & Humidity Read
  if (now - lastDHTMillis >= DHT_INTERVAL) {
    const float temperature = dht.readTemperature();
    const float humidity = dht.readHumidity();
    if (!isnan(temperature) && !isnan(humidity)) {
      lastTempC = temperature;
      lastHum = humidity;
    }
    lastDHTMillis = now;
  }
}

// ==============================================================================
// 11. NON-BLOCKING WI-FI RECONNECTION
// ==============================================================================
void handleWiFiReconnection(unsigned long now) {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  // If Wi-Fi is disconnected, attempt non-blocking reconnect periodically
  if (now - lastWiFiAttemptMillis >= WIFI_RETRY_INTERVAL) {
    lastWiFiAttemptMillis = now;
    Serial.println("[Wi-Fi] Attempting non-blocking reconnection...");
    WiFi.disconnect();
    WiFi.reconnect();
  }
}

// ==============================================================================
// 12. SEND JSON THROUGH WI-FI (HTTP POST to TX-PC)
// Exact JSON format with "machineHealth" from on-device Edge AI
// ==============================================================================
void sendSensorData() {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.setTimeout(1500); // Short timeout so networking never stalls loop

  // Construct valid JSON with numeric values and machineHealth string
  char jsonBuffer[256];
  snprintf(jsonBuffer, sizeof(jsonBuffer),
           "{\"machineId\":\"%s\","
           "\"temperature\":%.1f,"
           "\"humidity\":%.1f,"
           "\"vibrationX\":%.2f,"
           "\"vibrationY\":%.2f,"
           "\"vibrationZ\":%.2f,"
           "\"current\":%.2f,"
           "\"machineHealth\":\"%s\"}",
           machineId,
           isnan(lastTempC) ? 25.0f : lastTempC,
           isnan(lastHum) ? 50.0f : lastHum,
           lastAccelX,
           lastAccelY,
           lastAccelZ,
           lastCurrentA,
           machineHealthStr);

  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/json");
  const int httpCode = http.POST((uint8_t*)jsonBuffer, strlen(jsonBuffer));

  if (httpCode > 0) {
    Serial.printf("[TX-PC Sent] Code: %d | Health: %s\n", httpCode, machineHealthStr);
  } else {
    Serial.printf("[TX-PC Fail] Error: %s\n", http.errorToString(httpCode).c_str());
  }
  http.end();
}

// ==============================================================================
// 13. INNOVATIVE INDUSTRIAL TFT UI DESIGN (160 x 128)
// Zero flicker, color-coded, compact, and memory-efficient primitives
// ==============================================================================

void drawStaticLayout() {
  tft.fillScreen(COLOR_BG);

  // --- Header Bar (y: 0 to 18) ---
  tft.fillRect(0, 0, 160, 18, COLOR_BAR_BG);
  tft.drawFastHLine(0, 18, 160, COLOR_CYAN_ACC);

  tft.setTextColor(COLOR_WHITE_TEXT);
  tft.setTextSize(1);
  tft.setCursor(6, 5);
  tft.print(machineId);

  tft.setCursor(68, 5);
  tft.setTextColor(COLOR_MUTED_TEXT);
  tft.print("MONITOR");

  // AI Active badge
  tft.fillRect(116, 3, 40, 12, COLOR_CARD_BG);
  tft.drawRect(116, 3, 40, 12, aiAvailable ? COLOR_NORMAL : COLOR_CRITICAL);
  tft.setCursor(120, 5);
  tft.setTextColor(aiAvailable ? COLOR_NORMAL : COLOR_CRITICAL);
  tft.print(aiAvailable ? "AI:OK" : "AI:ERR");

  // --- Static Metric Labels & Icons (y: 22 to 62) ---
  // Temperature Row (y = 22)
  drawThermometerIcon(5, 23, COLOR_MUTED_TEXT);
  tft.setTextColor(COLOR_MUTED_TEXT);
  tft.setCursor(16, 23);
  tft.print("TEMP:");

  // Humidity Row (y = 36)
  drawDropletIcon(5, 37, COLOR_MUTED_TEXT);
  tft.setTextColor(COLOR_MUTED_TEXT);
  tft.setCursor(16, 37);
  tft.print("HUM :");

  // Current Row (y = 50)
  drawCurrentIcon(5, 51, COLOR_MUTED_TEXT);
  tft.setTextColor(COLOR_MUTED_TEXT);
  tft.setCursor(16, 51);
  tft.print("CURR:");

  // --- Vibration Card Section (y: 64 to 86) ---
  tft.drawFastHLine(4, 63, 152, COLOR_BORDER);
  drawVibrationIcon(5, 66, COLOR_CYAN_ACC);
  tft.setTextColor(COLOR_CYAN_ACC);
  tft.setCursor(16, 66);
  tft.print("VIBRATION (g)");

  tft.setTextColor(COLOR_MUTED_TEXT);
  tft.setCursor(6, 76);  tft.print("X:");
  tft.setCursor(58, 76); tft.print("Y:");
  tft.setCursor(110, 76);tft.print("Z:");

  // --- Machine Health AI Banner Section (y: 88 to 112) ---
  tft.drawFastHLine(4, 87, 152, COLOR_BORDER);

  // --- Footer Bar (y: 114 to 128) ---
  tft.fillRect(0, 114, 160, 14, COLOR_BAR_BG);
  tft.drawFastHLine(0, 113, 160, COLOR_BORDER);

  tft.setTextColor(COLOR_CYAN_ACC);
  tft.setCursor(110, 117);
  tft.print("EDGE AI");
}

void drawDynamicValues() {
  char buf[24];
  const bool wifiConnected = (WiFi.status() == WL_CONNECTED);

  // 1. Temperature Value (Update only if changed)
  const float currentTemp = isnan(lastTempC) ? 0.0f : lastTempC;
  if (abs(currentTemp - prevTemp) >= 0.1f) {
    prevTemp = currentTemp;
    uint16_t tempColor = COLOR_NORMAL;
    if (currentTemp >= 75.0f) tempColor = COLOR_CRITICAL;
    else if (currentTemp >= 60.0f) tempColor = COLOR_WARNING;

    tft.fillRect(52, 22, 102, 11, COLOR_BG);
    drawThermometerIcon(5, 23, tempColor);
    tft.setTextColor(tempColor);
    tft.setTextSize(1);
    snprintf(buf, sizeof(buf), "%5.1f C", currentTemp);
    tft.setCursor(52, 23);
    tft.print(buf);
  }

  // 2. Humidity Value
  const float currentHum = isnan(lastHum) ? 0.0f : lastHum;
  if (abs(currentHum - prevHum) >= 0.2f) {
    prevHum = currentHum;
    uint16_t humColor = COLOR_NORMAL;
    if (currentHum >= 75.0f) humColor = COLOR_CRITICAL;
    else if (currentHum >= 65.0f) humColor = COLOR_WARNING;

    tft.fillRect(52, 36, 102, 11, COLOR_BG);
    drawDropletIcon(5, 37, humColor);
    tft.setTextColor(humColor);
    tft.setTextSize(1);
    snprintf(buf, sizeof(buf), "%5.1f %%", currentHum);
    tft.setCursor(52, 37);
    tft.print(buf);
  }

  // 3. Current Value
  if (abs(lastCurrentA - prevCurrent) >= 0.05f) {
    prevCurrent = lastCurrentA;
    uint16_t currColor = COLOR_NORMAL;
    if (lastCurrentA >= 13.0f) currColor = COLOR_CRITICAL;
    else if (lastCurrentA >= 10.0f) currColor = COLOR_WARNING;

    tft.fillRect(52, 50, 102, 11, COLOR_BG);
    drawCurrentIcon(5, 51, currColor);
    tft.setTextColor(currColor);
    tft.setTextSize(1);
    snprintf(buf, sizeof(buf), "%5.2f A", lastCurrentA);
    tft.setCursor(52, 51);
    tft.print(buf);
  }

  // 4. Vibration X, Y, Z Values
  if (abs(lastAccelX - prevX) >= 0.02f || abs(lastAccelY - prevY) >= 0.02f || abs(lastAccelZ - prevZ) >= 0.02f) {
    prevX = lastAccelX; prevY = lastAccelY; prevZ = lastAccelZ;

    // X
    tft.fillRect(18, 76, 38, 10, COLOR_BG);
    uint16_t colX = (abs(lastAccelX) > 0.8f) ? COLOR_CRITICAL : (abs(lastAccelX) > 0.4f) ? COLOR_WARNING : COLOR_NORMAL;
    tft.setTextColor(colX);
    snprintf(buf, sizeof(buf), "%+.2f", lastAccelX);
    tft.setCursor(18, 76); tft.print(buf);

    // Y
    tft.fillRect(70, 76, 38, 10, COLOR_BG);
    uint16_t colY = (abs(lastAccelY) > 0.8f) ? COLOR_CRITICAL : (abs(lastAccelY) > 0.4f) ? COLOR_WARNING : COLOR_NORMAL;
    tft.setTextColor(colY);
    snprintf(buf, sizeof(buf), "%+.2f", lastAccelY);
    tft.setCursor(70, 76); tft.print(buf);

    // Z
    tft.fillRect(122, 76, 36, 10, COLOR_BG);
    uint16_t colZ = (abs(lastAccelZ - 1.0f) > 0.8f) ? COLOR_CRITICAL : (abs(lastAccelZ - 1.0f) > 0.4f) ? COLOR_WARNING : COLOR_NORMAL;
    tft.setTextColor(colZ);
    snprintf(buf, sizeof(buf), "%+.2f", lastAccelZ);
    tft.setCursor(122, 76); tft.print(buf);
  }

  // 5. Machine Health Banner (Redraw only when state changes)
  if (predictedClassIndex != prevHealthIdx) {
    prevHealthIdx = predictedClassIndex;

    uint16_t bannerBg = COLOR_CARD_BG;
    uint16_t bannerBorder = COLOR_NORMAL;
    uint16_t textColor = COLOR_NORMAL;
    const char* displayLabel = machineHealthStr;

    if (predictedClassIndex == 0) {
      // Critical
      bannerBg = 0x3800; // Deep red
      bannerBorder = COLOR_CRITICAL;
      textColor = COLOR_WHITE_TEXT;
    } else if (predictedClassIndex == 2) {
      // Warning
      bannerBg = 0x3200; // Deep amber
      bannerBorder = COLOR_WARNING;
      textColor = COLOR_WARNING;
    } else {
      // Normal
      bannerBg = 0x0260; // Deep green
      bannerBorder = COLOR_NORMAL;
      textColor = COLOR_NORMAL;
    }

    tft.fillRoundRect(4, 90, 152, 21, 3, bannerBg);
    tft.drawRoundRect(4, 90, 152, 21, 3, bannerBorder);

    tft.setTextSize(1);
    tft.setTextColor(textColor);

    // Centered Health Status Text with alert marker
    if (predictedClassIndex == 0) {
      tft.setCursor(14, 96);
      tft.print("[!] CRITICAL FAULT");
    } else if (predictedClassIndex == 2) {
      tft.setCursor(20, 96);
      tft.print("[~] HEALTH: WARNING");
    } else {
      tft.setCursor(22, 96);
      tft.print("[*] HEALTH: NORMAL");
    }
  }

  // 6. Wi-Fi Status Bar Indicator (Footer)
  if (wifiConnected != prevWiFiStatus) {
    prevWiFiStatus = wifiConnected;

    tft.fillRect(4, 115, 100, 12, COLOR_BAR_BG);
    drawWiFiIcon(6, 116, wifiConnected);

    tft.setTextSize(1);
    tft.setCursor(20, 117);
    if (wifiConnected) {
      tft.setTextColor(COLOR_NORMAL);
      tft.print("WiFi: ONLINE");
    } else {
      tft.setTextColor(COLOR_CRITICAL);
      tft.print("WiFi: OFFLINE");
    }
  }
}

// ==============================================================================
// 14. MEMORY-EFFICIENT GRAPHICAL PRIMITIVES (ICONS)
// ==============================================================================
void drawThermometerIcon(int16_t x, int16_t y, uint16_t color) {
  tft.drawFastVLine(x + 2, y, 7, color);
  tft.drawFastVLine(x + 3, y, 7, color);
  tft.fillCircle(x + 2, y + 7, 3, color);
}

void drawDropletIcon(int16_t x, int16_t y, uint16_t color) {
  tft.drawPixel(x + 2, y, color);
  tft.drawLine(x + 1, y + 1, x, y + 5, color);
  tft.drawLine(x + 3, y + 1, x + 4, y + 5, color);
  tft.fillCircle(x + 2, y + 6, 2, color);
}

void drawCurrentIcon(int16_t x, int16_t y, uint16_t color) {
  // Lightning bolt symbol
  tft.drawLine(x + 3, y, x + 1, y + 4, color);
  tft.drawFastHLine(x + 1, y + 4, 3, color);
  tft.drawLine(x + 4, y + 4, x + 1, y + 9, color);
}

void drawVibrationIcon(int16_t x, int16_t y, uint16_t color) {
  // Vibration wave
  tft.drawLine(x, y + 4, x + 2, y + 1, color);
  tft.drawLine(x + 2, y + 1, x + 4, y + 7, color);
  tft.drawLine(x + 4, y + 7, x + 6, y + 4, color);
}

void drawWiFiIcon(int16_t x, int16_t y, bool connected) {
  uint16_t color = connected ? COLOR_NORMAL : COLOR_CRITICAL;
  tft.fillCircle(x + 4, y + 8, 1, color);
  tft.drawCircle(x + 4, y + 8, 4, color);
  if (!connected) {
    // Small cross over icon
    tft.drawLine(x + 1, y + 1, x + 7, y + 7, COLOR_CRITICAL);
  }
}
