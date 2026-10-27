class PIDController:
    def __init__(self, kp=0.4, ki=0.1, kd=0.4, max_integral=10.0, max_output=30.0, threshold=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.max_output = max_output
        self.threshold = threshold
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_output = 0.0
        self.prev_d = 0.0

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_output = 0.0
        self.prev_d = 0.0

    def clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def calculate(self, target_value, current_value):
        error = target_value - current_value

        if abs(error) <= self.threshold:
            self.reset()
            return 0.0

        derivative = error - self.prev_error
        self.integral += error
        self.integral = self.clamp(self.integral, -self.max_integral, self.max_integral)

        total = self.calculate_with_values(error, derivative, self.integral)
        total = self.clamp(total, -self.max_output, self.max_output)

        self.prev_error = error
        self.last_output = total

        return total

    def calculate_with_values(self, error, derivative: float, integral: float):
        p_value = error * self.kp
        i_value = self.ki * integral
        d_value = derivative * self.kd

        self.prev_d = d_value

        return p_value + i_value + d_value

    def predict(self):
        value = self.last_output + self.prev_d
        self.last_output = value

        return value
    