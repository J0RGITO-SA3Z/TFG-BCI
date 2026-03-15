import time
import csv
import threading
import winsound

class BEEP_interpreter:
    def __init__(self, pipe, restTime = 6, interval = 4, waitTime = 1, bestOf = 2):
        self.pipe = pipe

        self.predictions = []
        self.samples = []
        self.delays = []

        self.running = False
        self.threadBEEP = None
        self.threadListen = None

        # Tiempo de espera entre el pitido y la predicción
        self.restTime = restTime
        self.interval = interval
        self.waitTime = waitTime

        #self.ultimaPrediccion = None
        #self.timeStampUltimaPrediccion = None
        self.timeStampLastBeep = None
        self.actualPredictions = None
        self.bestOf = bestOf

        self.rachaContinua = True

    def _BEEP(self):
        while self.running:
            # TIEMPO DE DESCANSO
            time.sleep(self.restTime)

            i = 0
            self.rachaContinua = True

            while (i < self.bestOf) and self.rachaContinua:
                # SONAR PITIDO DE INICIO
                winsound.Beep(1000, 500) # freq, duracion

                # TOMAR TIEMPO
                self.timeStampLastBeep = time.perf_counter()

                # METER DELAY
                time.sleep(self.interval)

                # SONAR PITIDO DE FIN
                winsound.Beep(500, 500)

                # TIEMPO DE ESPERA
                time.sleep(self.waitTime)
                
                # INCREMENTAMOS BUCLE
                i+=1

    def _listen(self):
        while self.running:
            # RECIBIR
            if self.pipe.poll():
                prediction, last_sample, last_timestamp = self.pipe.recv()

                # COMPROBAR que ha pasado el tiempo suficiente
                if last_timestamp - self.timeStampLastBeep >= self.interval:
                    now = time.perf_counter()
                    delay = now - last_timestamp

                    if self.actualPredictions == None:
                        self.actualPredictions = prediction
                    elif self.actualPredictions != prediction:
                        self.rachaContinua = False

                    self.predictions.append(prediction)
                    self.samples.append(last_sample)
                    self.delays.append(delay)



    def start(self):
        self.running = True
        self.threadListen = threading.Thread(target=self._listen, daemon=True)
        self.threadBEEP = threading.Thread(target=self._BEEP, daemon=True)
        self.threadListen.start()
        self.threadBEEP.start()

    def stop(self):
        self.running = False
        if self.threadListen is not None:
            self.threadListen.join()
        if self.threadBEEP is not None:
            self.threadBEEP.join()
        

    def save(self, filename="predictions_log.csv"):

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["prediction", "sample", "delay"])

            for p, s, d in zip(self.predictions, self.samples, self.delays):
                writer.writerow([p, s, d])