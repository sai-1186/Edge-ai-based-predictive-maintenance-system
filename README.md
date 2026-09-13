# Edge AI Predictive Maintenance System

A complete full-stack industrial predictive maintenance system. It monitors industrial machine health using an **ESP32 Edge AI sensor node**, a **TX-PC** (communication/forwarding server), and an **RX-PC** (real-time monitoring dashboard) operating in a local network without requiring cloud services.

## System Architecture

```
Sensors (DHT11, MPU6050, ACS712)
       │
       ▼
ESP32 Sensor Node (Edge AI Device)
  ├── Temperature, Humidity, Current, Vibration Acquisition
  ├── StandardScaler Preprocessing
  ├── TensorFlow Lite Micro On-Chip Inference (predmodel.h)
  ├── Machine Health Classification ("Normal", "Warning", "Critical")
  ├── Industrial ST7735 TFT Color-Coded Display
  └── Wi-Fi JSON Dispatch (Non-blocking)
       │ (HTTP POST / WebSocket)
       ▼
TX-PC Forwarding Server (Node.js Network Broker)
  ├── Receives ESP32 JSON
  ├── Preserves ESP32 "machineHealth"
  └── Broadcasts to Dashboard via WebSockets
       │ (WebSockets)
       ▼
RX-PC Dashboard (React / Vite Real-Time UI)
  ├── Live Sensor Gauges & Real-Time Waveform Charts
  └── Machine Health Status & Predictive Maintenance Alerts
```

## Features
- **True Edge AI:** TensorFlow Lite Micro running directly on the ESP32 microcontroller with embedded neural network model weights (`predmodel.h`) and class labels (`predlabel.h`).
- **Resilient Non-Blocking Operation:** Local sensor reading, AI inference, and TFT display continue uninterrupted even if Wi-Fi disconnects or is unavailable.
- **Innovative TFT UI:** Compact industrial display (160x128 ST7735) with custom graphical primitive icons, color-coded health states, and zero-flicker partial updates.
- **Real-Time Monitoring Pipeline:** Ultra-low latency data transmission via WebSockets.
- **Simulation Mode:** Built-in data simulator for testing the UI and alerts without physical hardware connected.

---

## Installation & Setup

### 1. ESP32 (Sensor Node & Edge AI)

1. Open `esp32_sensor_node/sensor_transmitter/sensor_transmitter.ino` in the Arduino IDE.
2. Ensure the required libraries are installed (via Arduino Library Manager):
   - `TensorFlowLite_ESP32`
   - `Adafruit GFX Library`
   - `Adafruit ST7735 and ST7789 Library`
   - `Adafruit MPU6050`
   - `Adafruit Unified Sensor`
   - `DHT sensor library`
3. Update the Wi-Fi credentials and TX-PC server URL in `sensor_transmitter.ino`:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* serverUrl = "http://192.168.1.100:8080/api/data";
   ```
4. Select board **ESP32 Dev Module**, choose the correct COM port, and click **Upload**.

### 2. TX-PC (Communication & Forwarding Server)

This server receives data from the ESP32 and broadcasts it to the dashboard. It does **not** perform AI prediction; it serves strictly as the network forwarding layer.

1. Ensure Node.js (v18+) is installed.
2. Open a terminal and navigate to the `tx-server` directory:
   ```bash
   cd tx-server
   ```
3. Install dependencies:
   ```bash
   npm install
   ```
4. Start the server:
   ```bash
   node server.js
   ```
5. Check your TX-PC IP address (e.g. `ipconfig` on Windows).

### 3. RX-PC (Dashboard)

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Create or update `.env` with your TX-PC IP:
   ```env
   VITE_TX_SERVER_URL=ws://192.168.1.100:8080
   ```
   *(Use `ws://127.0.0.1:8080` if running locally on the same computer)*
4. Start the dashboard:
   ```bash
   npm run dev
   ```
5. Open your browser to `http://localhost:5173`.

---

### Example JSON Payload (ESP32 -> TX-PC -> RX-PC)

```json
{
  "machineId": "MOTOR-01",
  "temperature": 72.5,
  "humidity": 48.2,
  "vibrationX": 0.21,
  "vibrationY": 0.18,
  "vibrationZ": 0.35,
  "current": 8.6,
  "machineHealth": "Warning"
}
```
