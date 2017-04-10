import time
import sys
import math

class ProgressBar:
    start_time = None
    percent = 0.0
    steps = 0
    max_steps = None
    undersample = 1
    unit_prefix = {3:'k', 6:'M', 9:'G'}
    name = 'Loops'
    initial_run = True
    remaining = 0
    alpha = 0.2

    def __init__(self, max_steps, undersample=1, name='Loops'):
        self.undersample = undersample
        self.start_time = time.time()
        self.max_steps = max_steps
        self.steps = 0
        self.percent = self.steps / self.max_steps
        self.name = name

    def draw_progress_bar(self, steps=-1):
        self.steps = steps if steps > 0 else self.steps + 1
        if self.steps < self.max_steps and self.undersample > 1 and self.steps % self.undersample:
            return
        self.percent = self.steps / self.max_steps
        sys.stdout.write("\r")
        progress = ""
        for i in range(37):
            if i < int(37 * self.percent):
                progress += "="
            elif i == int(37 * self.percent):
                progress += ">"
            else:
                progress += " "

        elapsedTime = int(time.time() - self.start_time);
        if self.initial_run:
            self.remaining = int(elapsedTime * (1.0/self.percent) - elapsedTime)
            self.initial_run = False
        else:
            # exponential smoothing with weighting of new estimation of alpha
            self.remaining = self.alpha*(int(elapsedTime * (1.0/self.percent) - elapsedTime)) + (1-self.alpha)*self.remaining

        if (self.steps + self.undersample - 1 > self.max_steps):
            print("{:>6} {:8} {:s} [{:s}] {:2d}% ETA Done!   ".format(
                self.unit_prefix_format(self.steps), self.name, self.format_time(elapsedTime),
                progress, int(self.percent * 100), self.format_time(self.remaining)))
        else:
            sys.stdout.write("{:>6} {:8} {:s} [{:s}] {:2d}% ETA {:} ".format(
                self.unit_prefix_format(self.steps), self.name, self.format_time(elapsedTime),
                progress, int(self.percent * 100), self.format_time(self.remaining)))
            sys.stdout.flush()

    def format_time(self, seconds):
        return "{:d}:{:02d}:{:02d}".format(int(seconds/3600), int((seconds%3600)/60), int((seconds%3600))%60)

    def unit_prefix_format(self, n):
        str = ""
        if n < 1000:
            str = "{:5d}".format(n)
        else:
            lg = int(math.log10(n))
            prefix = lg - lg%3
            prefmt = "{:2.1f}{:s}" if (lg%3 < 2) else "{:3.0f}{:s}"
            str = prefmt.format(n/(10**prefix), self.unit_prefix[prefix])
        return str

#
# 2.5M 0:02:49 [14.7k/s] [====================================>] 40% ETA 1:05:38
#24.2GB 0:44:21 [4.45MB/s] [=============>                     ] 40% ETA 1:05:38
#98.0k Tweeds   0:00:06 [=>                                   ]  3% ETA 0:02:31