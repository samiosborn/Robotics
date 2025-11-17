# src/config/imu_constants.py

# --- Sensitivity Tables ---
# Accel Sensitivity
ACCEL_SENSITIVITY_LSB_PER_G = {
    0: 16384.0,  # FS_SEL=0 is +/- 2  g
    1: 8192.0,   # FS_SEL=1 is +/- 4  g
    2: 4096.0,   # FS_SEL=2 is +/- 8  g
    3: 2048.0,   # FS_SEL=3 is +/- 16 g
}
# Gyro Sensitivity
GYRO_SENSITIVITY_LSB_PER_DPS = {
    0: 131.0,   # FS_SEL=0 is +/- 250  dps
    1: 65.5,    # FS_SEL=1 is +/- 500  dps
    2: 32.8,    # FS_SEL=2 is +/- 1000 dps
    3: 16.4,    # FS_SEL=3 is +/- 2000 dps
}

# --- Physics ---
# Gravitational field strength (m/s^2)
GRAVITY_M_S2 = 9.80665

