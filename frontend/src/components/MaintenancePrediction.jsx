import React from 'react';
import { Cpu, Wrench } from 'lucide-react';

const MaintenancePrediction = ({ latestData }) => {
  if (!latestData) return null;

  const { healthScore, riskLevel, recommendation, status } = latestData;

  let riskColor = "text-accent-green";
  if (riskLevel === "HIGH") riskColor = "text-accent-red";
  else if (riskLevel === "MEDIUM") riskColor = "text-accent-yellow";

  return (
    <div className="bg-slate-800 p-5 rounded-xl border border-slate-700 h-full flex flex-col">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-slate-400 font-medium text-sm tracking-wider uppercase flex items-center gap-2">
          <Cpu className="h-4 w-4" />
          AI Maintenance Prediction
        </h3>
      </div>

      <div className="flex flex-col gap-6 flex-1">
        <div className="flex items-center gap-6">
          <div className="relative">
            <svg className="w-24 h-24 transform -rotate-90">
              <circle cx="48" cy="48" r="40" stroke="#1e293b" strokeWidth="8" fill="none" />
              <circle 
                cx="48" cy="48" r="40" 
                stroke={healthScore > 80 ? '#22c55e' : healthScore > 50 ? '#eab308' : '#ef4444'} 
                strokeWidth="8" 
                fill="none" 
                strokeDasharray="251.2" 
                strokeDashoffset={251.2 - (251.2 * healthScore) / 100}
                className="transition-all duration-1000 ease-out"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-2xl font-bold text-slate-200">{healthScore}%</span>
            </div>
          </div>
          
          <div className="flex-1">
            <div className="mb-2">
              <div className="text-slate-500 text-xs font-bold mb-1 uppercase">Machine</div>
              <div className="text-lg text-slate-200 font-medium">{latestData.machineId}</div>
            </div>
            <div className="flex gap-4">
              <div>
                <div className="text-slate-500 text-xs font-bold mb-1 uppercase">Machine Health</div>
                <div className={`text-lg font-bold ${riskColor}`}>{latestData.machineHealth || latestData.status || "NORMAL"}</div>
              </div>
              <div>
                <div className="text-slate-500 text-xs font-bold mb-1 uppercase">Risk Level</div>
                <div className={`text-lg font-bold ${riskColor}`}>{riskLevel}</div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-slate-900 rounded-lg p-4 border border-slate-700 flex-1 flex flex-col justify-center">
          <div className="flex items-start gap-3">
            <div className={`p-2 rounded-full bg-slate-800 border ${riskLevel === 'HIGH' ? 'border-accent-red text-accent-red' : riskLevel === 'MEDIUM' ? 'border-accent-yellow text-accent-yellow' : 'border-accent-green text-accent-green'}`}>
              <Wrench className="h-5 w-5" />
            </div>
            <div>
              <div className="text-slate-500 text-xs font-bold mb-1 uppercase">Recommendation</div>
              <div className="text-slate-200 text-sm">{recommendation}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MaintenancePrediction;
