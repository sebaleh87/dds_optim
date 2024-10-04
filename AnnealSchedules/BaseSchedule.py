class BaseScheduleClass:
    def __init__(self, AnnealConfig):
        self.start_temperature = AnnealConfig["T_start"]

    def update_temp(self):
        return self.start_temperature

# Example usage:
# schedule = BaseScheduleClass(100)
# schedule.update_temp(90)
# print(schedule.temperature)  # Output: 90