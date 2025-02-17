class LogFileCreator:
    def __init__(self, file_name):
        #self.output_path = f"D:/PWR/Praca magisterska/Log/{file_name}.txt"
        self.output_path = f"/mnt/d/PWR/Praca magisterska/Log/{file_name}.txt"
        self.string_spit_stars = 60 * "x"
        self.string_spit_tilds = 60*"~"
        self.string_spit_eq = 60*"="
        self.string_spit_x = 60* "x"


    def print_and_write_log(self, string):
        print(string)
        f = open(self.output_path, "a", encoding="utf-8")
        f.write(string+"\n")
        f.close()
    def count_time(self, time_start, time_end):
        return time_end - time_start