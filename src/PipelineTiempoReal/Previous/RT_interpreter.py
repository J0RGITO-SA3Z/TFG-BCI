import time
import csv
import threading

class RT_interpreter:
    def __init__(self, pipe):
        self.pipe = pipe

        self.predictions = []
        self.samples = []
        self.delays = []

        self.running = False
        self.thread = None

    def _listen(self):
        while self.running:
            if self.pipe.poll():
                prediction, last_sample, last_timestamp = self.pipe.recv()

                now = time.perf_counter()
                delay = now - last_timestamp

                self.predictions.append(prediction)
                self.samples.append(last_sample)
                self.delays.append(delay)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def save(self, filename="predictions_log.csv"):

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["prediction", "sample", "delay"])

            for p, s, d in zip(self.predictions, self.samples, self.delays):
                writer.writerow([p, s, d])