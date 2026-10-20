import random
from datetime import datetime

# Sensor Safe Thresholds
THRESHOLDS = {
    "temperature_max": 55.0,  # °C (Extreme rail temperature warning)
    "vibration_max": 2.5,     # g (Excessive track vibration warning)
    "stress_max": 300.0,      # MPa (High mechanical rail stress)
    "humidity_max": 90.0      # % (High moisture / corrosion risk)
}

class IoTTelemetryService:
    def __init__(self):
        self.battery_level = 88.5
        self.solar_charging = True

    def generate_simulated_readings(self, simulate_fault: bool = False) -> dict:
        """
        Generates realistic IoT sensor data for railway tracks.
        """
        # Drain battery slightly
        self.battery_level = max(10.0, self.battery_level - random.uniform(0.05, 0.2))

        if simulate_fault or random.random() < 0.15:
            # Simulate anomalous spike
            temp = round(random.uniform(52.0, 68.0), 1)
            vib = round(random.uniform(2.3, 4.2), 2)
            stress = round(random.uniform(280.0, 360.0), 1)
            humidity = round(random.uniform(60.0, 92.0), 1)
        else:
            # Normal operating range
            temp = round(random.uniform(22.0, 42.0), 1)
            vib = round(random.uniform(0.2, 1.4), 2)
            stress = round(random.uniform(110.0, 240.0), 1)
            humidity = round(random.uniform(40.0, 75.0), 1)

        # Check threshold alerts
        alerts = []
        status = "NORMAL"

        if temp > THRESHOLDS["temperature_max"]:
            alerts.append(f"CRITICAL: High Rail Temperature ({temp}°C > {THRESHOLDS['temperature_max']}°C)")
            status = "CRITICAL"
        if vib > THRESHOLDS["vibration_max"]:
            alerts.append(f"WARNING: High Track Vibration ({vib}g > {THRESHOLDS['vibration_max']}g)")
            if status != "CRITICAL": status = "WARNING"
        if stress > THRESHOLDS["stress_max"]:
            alerts.append(f"CRITICAL: Excessive Rail Stress ({stress}MPa > {THRESHOLDS['stress_max']}MPa)")
            status = "CRITICAL"
        if humidity > THRESHOLDS["humidity_max"]:
            alerts.append(f"WARNING: Elevated Track Moisture ({humidity}% > {THRESHOLDS['humidity_max']}%)")
            if status == "NORMAL": status = "WARNING"

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temp,
            "vibration": vib,
            "stress": stress,
            "humidity": humidity,
            "battery_level": round(self.battery_level, 1),
            "status_alert": status,
            "alerts": alerts,
            "thresholds": THRESHOLDS,
            "iot_hardware": ["ESP32-WROOM-32", "Raspberry Pi 4", "MQTT Broker (mosquitto)", "REST API Gateways"]
        }

iot_engine = IoTTelemetryService()
