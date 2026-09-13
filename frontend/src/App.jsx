import React, { useState, useEffect, useRef } from 'react';
import { Settings, Power, Activity, Thermometer, Droplets, Zap, ShieldCheck } from 'lucide-react';
import StatusCard from './components/StatusCard';
import RealTimeChart from './charts/RealTimeChart';
import VibrationAnalysis from './components/VibrationAnalysis';
import MaintenancePrediction from './components/MaintenancePrediction';
import AlertSystem from './components/AlertSystem';

const WS_URL = import.meta.env.VITE_TX_SERVER_URL || 'ws://127.0.0.1:8080';
const HTTP_URL = WS_URL.replace('ws://', 'http://').replace('wss://', 'https://');
const MAX_DATA_POINTS = 50;

const generateInitialDummyData = () => {
  const dummyHistory = [];
  let currentTime = new Date();
  
  for (let i = MAX_DATA_POINTS - 1; i >= 0; i--) {
    let t = new Date(currentTime.getTime() - i * 2000);
    dummyHistory.push({
      time: t.toLocaleTimeString([], { hour12: false }),
      temp: parseFloat((65 + Math.random() * 5).toFixed(1)),
      humidity: parseFloat((45 + Math.random() * 5).toFixed(1)),
      current: parseFloat((7 + Math.random() * 1).toFixed(1)),
      z: parseFloat((0.2 + Math.random() * 0.1).toFixed(2)),
      x: parseFloat((0.2 + Math.random() * 0.1).toFixed(2)),
      y: parseFloat((0.2 + Math.random() * 0.1).toFixed(2)),
    });
  }
  return dummyHistory;
};

const initialDummyData = generateInitialDummyData();

const initialLatestData = {
  machineId: "MOTOR-01",
  temperature: initialDummyData[49].temp,
  humidity: initialDummyData[49].humidity,
  current: initialDummyData[49].current,
  vibrationX: initialDummyData[49].x,
  vibrationY: initialDummyData[49].y,
  vibrationZ: initialDummyData[49].z,
  vibrationRMS: 0.25,
  healthScore: 98,
  status: "HEALTHY",
  riskLevel: "LOW",
  recommendation: "System operating normally"
};

const initialAlerts = [
  {
    timestamp: new Date().toISOString(),
    severity: "INFO",
    message: "System initialized with local dummy data.",
    parameter: "Status",
    value: "OK"
  }
];

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [simMode, setSimMode] = useState(false);
  const [latestData, setLatestData] = useState(initialLatestData);
  
  const [tempHistory, setTempHistory] = useState(initialDummyData.map(d => ({ time: d.time, temp: d.temp })));
  const [humidityHistory, setHumidityHistory] = useState(initialDummyData.map(d => ({ time: d.time, humidity: d.humidity })));
  const [currentHistory, setCurrentHistory] = useState(initialDummyData.map(d => ({ time: d.time, current: d.current })));
  const [vibrationHistory, setVibrationHistory] = useState(initialDummyData.map(d => ({ time: d.time, x: d.x, y: d.y, z: d.z })));
  const [alerts, setAlerts] = useState(initialAlerts);
  
  const ws = useRef(null);

  // Initialize and check simulation status
  useEffect(() => {
    fetch(`${HTTP_URL}/api/simulation/status`)
      .then(res => res.json())
      .then(data => setSimMode(data.enabled))
      .catch(err => console.log('Could not fetch sim status', err));
  }, []);

  useEffect(() => {
    const connectWebSocket = () => {
      ws.current = new WebSocket(WS_URL);

      ws.current.onopen = () => {
        setIsConnected(true);
      };

      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        const timeStr = new Date(data.timestamp).toLocaleTimeString([], { hour12: false });
        
        setLatestData(data);
        
        // Update Histories
        setTempHistory(prev => [...prev.slice(-MAX_DATA_POINTS + 1), { time: timeStr, temp: data.temperature }]);
        setHumidityHistory(prev => [...prev.slice(-MAX_DATA_POINTS + 1), { time: timeStr, humidity: data.humidity }]);
        setCurrentHistory(prev => [...prev.slice(-MAX_DATA_POINTS + 1), { time: timeStr, current: data.current }]);
        setVibrationHistory(prev => [...prev.slice(-MAX_DATA_POINTS + 1), { time: timeStr, x: data.vibrationX, y: data.vibrationY, z: data.vibrationZ }]);

        // Manage Alerts based on Status
        if (data.status !== "HEALTHY") {
          setAlerts(prev => {
            const newAlert = {
              timestamp: data.timestamp,
              severity: data.status,
              message: data.recommendation,
              parameter: "AI Risk Level",
              value: data.riskLevel
            };
            // Prevent spamming identical alerts
            if (prev.length > 0 && prev[0].message === newAlert.message && prev[0].severity === newAlert.severity) {
              return prev;
            }
            return [newAlert, ...prev].slice(0, 50);
          });
        }
      };

      ws.current.onclose = () => {
        setIsConnected(false);
        // Attempt to reconnect every 3 seconds
        setTimeout(connectWebSocket, 3000);
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket Error:', error);
        ws.current.close();
      };
    };

    connectWebSocket();

    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  const toggleSimulation = async () => {
    try {
      const endpoint = simMode ? '/api/simulation/stop' : '/api/simulation/start';
      await fetch(`${HTTP_URL}${endpoint}`, { method: 'POST' });
      setSimMode(!simMode);
    } catch (error) {
      console.error('Error toggling simulation:', error);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200 p-4 md:p-6 font-sans">
      
      {/* HEADER */}
      <header className="flex flex-col md:flex-row justify-between items-center mb-8 bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-sm">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <Activity className="text-accent-blue" /> EDGE AI PREDICTIVE MAINTENANCE
          </h1>
          <p className="text-slate-400 text-sm mt-1">Real-Time Industrial Machine Monitoring</p>
        </div>
        
        <div className="flex items-center gap-6 mt-4 md:mt-0">
          <div className="flex flex-col">
            <span className="text-xs text-slate-500 uppercase font-bold tracking-wider">Machine ID</span>
            <span className="font-mono font-medium text-slate-200">{latestData?.machineId || "WAITING..."}</span>
          </div>

          <div className="flex flex-col">
            <span className="text-xs text-slate-500 uppercase font-bold tracking-wider">Data source</span>
            <span className={`font-mono text-xs font-bold ${latestData?.source === 'SENSOR' ? 'text-accent-green' : 'text-amber-400'}`}>
              {latestData?.source || 'WAITING'}
            </span>
          </div>
          
          <div className="flex items-center gap-2 bg-slate-900 py-1.5 px-3 rounded-full border border-slate-700">
            <div className={`h-2.5 w-2.5 rounded-full ${isConnected ? 'bg-accent-green shadow-[0_0_8px_rgba(34,197,94,0.6)]' : 'bg-slate-500'}`}></div>
            <span className={`text-xs font-bold tracking-wider ${isConnected ? 'text-accent-green' : 'text-slate-500'}`}>
              {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
            </span>
          </div>
          
          <button 
            onClick={toggleSimulation}
            className={`flex items-center gap-2 py-1.5 px-3 rounded text-xs font-bold tracking-wider transition-colors border ${
              simMode 
                ? 'bg-accent-blue/20 text-accent-blue border-accent-blue/30 hover:bg-accent-blue/30' 
                : 'bg-slate-700 text-slate-400 border-slate-600 hover:bg-slate-600'
            }`}
          >
            <Power size={14} /> SIMULATION {simMode ? 'ON' : 'OFF'}
          </button>
        </div>
      </header>

      {/* TOP STATUS CARDS */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
        <StatusCard 
          title="Machine Health" 
          value={latestData?.machineHealth || (latestData ? `${latestData.healthScore}%` : '--')} 
          unit={latestData?.healthScore !== undefined ? `(${latestData.healthScore}%)` : ''}
          status={latestData?.machineHealth ? latestData.machineHealth.toUpperCase() : (latestData?.status || "UNKNOWN")}
          icon={ShieldCheck}
        />
        <StatusCard 
          title="Temperature" 
          value={latestData?.temperature || '--'} 
          unit="°C"
          status={latestData?.temperature > 85 ? "CRITICAL" : latestData?.temperature > 70 ? "WARNING" : "NORMAL"}
          icon={Thermometer}
        />
        <StatusCard 
          title="Humidity" 
          value={latestData?.humidity || '--'} 
          unit="%"
          status={latestData?.humidity > 75 ? "CRITICAL" : latestData?.humidity > 60 ? "WARNING" : "NORMAL"}
          icon={Droplets}
        />
        <StatusCard 
          title="Vibration (RMS)" 
          value={latestData?.vibrationRMS || '--'} 
          unit="g"
          status={latestData?.vibrationRMS > 0.8 ? "CRITICAL" : latestData?.vibrationRMS > 0.4 ? "WARNING" : "NORMAL"}
          icon={Activity}
        />
        <StatusCard 
          title="Current" 
          value={latestData?.current || '--'} 
          unit="A"
          status={latestData?.current > 13 ? "CRITICAL" : latestData?.current > 10 ? "WARNING" : "NORMAL"}
          icon={Zap}
        />
      </div>

      {/* MAIN GRID LAYOUT */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* LEFT COLUMN: CHARTS */}
        <div className="lg:col-span-2 space-y-6">
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <RealTimeChart 
              name="Temperature" 
              unit="°C"
              data={tempHistory} 
              dataKey="temp" 
              color="#ef4444" 
            />
            <RealTimeChart 
              name="Current" 
              unit="A"
              data={currentHistory} 
              dataKey="current" 
              color="#eab308" 
            />
          </div>
          
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
             <RealTimeChart 
              name="Vibration Z" 
              unit="g"
              data={vibrationHistory} 
              dataKey="z" 
              color="#3b82f6" 
            />
             <RealTimeChart 
              name="Humidity" 
              unit="%"
              data={humidityHistory} 
              dataKey="humidity" 
              color="#22c55e" 
            />
          </div>
        </div>

        {/* RIGHT COLUMN: AI & ANALYSIS */}
        <div className="space-y-6 flex flex-col">
          <div className="h-[300px]">
            <VibrationAnalysis latestData={latestData} />
          </div>
          <div className="h-[250px]">
            <MaintenancePrediction latestData={latestData} />
          </div>
          <div className="flex-1 min-h-[300px]">
            <AlertSystem alerts={alerts} />
          </div>
        </div>
        
      </div>
    </div>
  );
}

export default App;
