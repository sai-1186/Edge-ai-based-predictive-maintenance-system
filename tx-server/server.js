const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const cors = require('cors');
const { generateSimulatedData } = require('./sensors/simulator');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(cors());
app.use(express.json());

const PORT = 8080;

let isSimulationEnabled = false;
let simulationInterval = null;

// Track active WebSocket clients
wss.on('connection', (ws) => {
  console.log('Client connected to WebSocket (RX-PC / Dashboard)');

  // Also support receiving JSON directly over WebSocket from ESP32 if configured
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message.toString());
      if (data && data.machineId) {
        broadcastData(data, 'ESP32_EDGE_AI');
      }
    } catch (err) {
      console.warn('Received non-JSON WebSocket message:', message.toString());
    }
  });

  ws.on('close', () => {
    console.log('Client disconnected from WebSocket');
  });
});

/**
 * Forward data to all connected WebSocket clients (RX-PC / Dashboard)
 * TX-PC does NOT calculate or alter machineHealth.
 * machineHealth is generated directly by the ESP32 TensorFlow Lite model.
 */
function broadcastData(data, source = 'ESP32_EDGE_AI') {
  const machineHealth = data.machineHealth || 'Normal';

  // Compute vibration RMS for UI visualization if not already provided
  const vx = Number(data.vibrationX) || 0;
  const vy = Number(data.vibrationY) || 0;
  const vz = Number(data.vibrationZ) || 0;
  const vibrationRMS = data.vibrationRMS || Math.sqrt((vx * vx + vy * vy + vz * vz) / 3).toFixed(2);

  // Map machineHealth from ESP32 Edge AI directly into UI presentation helpers
  let status = 'HEALTHY';
  let riskLevel = 'LOW';
  let healthScore = 98;
  let recommendation = 'ESP32 Edge AI: Operating normally.';

  if (machineHealth.toLowerCase() === 'critical' || machineHealth.toLowerCase() === 'fault') {
    status = 'CRITICAL';
    riskLevel = 'HIGH';
    healthScore = 25;
    recommendation = 'ESP32 Edge AI: Critical motor fault predicted!';
  } else if (machineHealth.toLowerCase() === 'warning' || machineHealth.toLowerCase() === 'warming') {
    status = 'WARNING';
    riskLevel = 'MEDIUM';
    healthScore = 65;
    recommendation = 'ESP32 Edge AI: Warning condition detected.';
  }

  const payloadObject = {
    ...data,
    source,
    machineHealth,
    status: data.status || status,
    riskLevel: data.riskLevel || riskLevel,
    healthScore: data.healthScore !== undefined ? data.healthScore : healthScore,
    recommendation: data.recommendation || recommendation,
    vibrationRMS,
    timestamp: data.timestamp || new Date().toISOString()
  };

  const payload = JSON.stringify(payloadObject);

  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(payload);
    }
  });

  return payloadObject;
}

// HTTP POST endpoint for ESP32 to push sensor data + edge prediction
app.post('/api/data', (req, res) => {
  if (isSimulationEnabled) {
    return res.status(400).json({ error: 'Simulation mode is currently active' });
  }

  try {
    const data = req.body;
    const requiredFields = [
      'temperature', 'humidity', 'vibrationX', 'vibrationY', 'vibrationZ', 'current'
    ];

    if (!data.machineId || requiredFields.some((field) => !Number.isFinite(data[field]))) {
      return res.status(400).json({ error: 'Invalid sensor data format' });
    }

    console.log(`[ESP32 -> TX-PC] Machine: ${data.machineId} | Health: ${data.machineHealth} | Temp: ${data.temperature}°C | Cur: ${data.current}A`);
    const forwarded = broadcastData(data, 'ESP32_EDGE_AI');
    res.status(200).json({ success: true, machineHealth: forwarded.machineHealth });
  } catch (error) {
    console.error('Error handling sensor data:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Simulation Control Endpoints (for testing without physical hardware)
app.post('/api/simulation/start', (req, res) => {
  if (isSimulationEnabled) {
    return res.json({ status: 'already running' });
  }

  isSimulationEnabled = true;
  console.log('Simulation mode started');

  simulationInterval = setInterval(() => {
    const simData = generateSimulatedData();
    broadcastData(simData, 'SIMULATION');
  }, 2000);

  res.json({ status: 'started' });
});

app.post('/api/simulation/stop', (req, res) => {
  if (!isSimulationEnabled) {
    return res.json({ status: 'already stopped' });
  }

  isSimulationEnabled = false;
  if (simulationInterval) clearInterval(simulationInterval);
  console.log('Simulation mode stopped');

  res.json({ status: 'stopped' });
});

app.get('/api/simulation/status', (req, res) => {
  res.json({ enabled: isSimulationEnabled });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`TX-PC Forwarding Server running on http://0.0.0.0:${PORT}`);
  console.log(`WebSocket Server listening on ws://0.0.0.0:${PORT}`);
});
