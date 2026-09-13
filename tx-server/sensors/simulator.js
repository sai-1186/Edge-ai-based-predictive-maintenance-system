function generateSimulatedData(machineId = "MOTOR-01") {
  // Base normal values
  let temp = 40 + Math.random() * 10;
  let humidity = 45 + Math.random() * 15;
  let vx = 0.05 + Math.random() * 0.15;
  let vy = 0.05 + Math.random() * 0.15;
  let vz = 0.90 + Math.random() * 0.15;
  let current = 6.0 + Math.random() * 2.5;
  let machineHealth = "Normal";

  // Simulate occasional machine state changes
  const rand = Math.random();
  if (rand > 0.85) {
    // Critical anomaly
    temp += 35;
    vx += 1.2;
    vz += 1.0;
    current += 6.5;
    machineHealth = "Critical";
  } else if (rand > 0.65) {
    // Warning state
    temp += 20;
    vx += 0.35;
    current += 3.0;
    machineHealth = "Warning";
  }

  return {
    machineId,
    temperature: parseFloat(temp.toFixed(1)),
    humidity: parseFloat(humidity.toFixed(1)),
    vibrationX: parseFloat(vx.toFixed(2)),
    vibrationY: parseFloat(vy.toFixed(2)),
    vibrationZ: parseFloat(vz.toFixed(2)),
    current: parseFloat(current.toFixed(1)),
    machineHealth
  };
}

module.exports = { generateSimulatedData };
