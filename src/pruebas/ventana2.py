import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

NUM_CHANNELS = 16
SAMPLES_PER_CHANNEL = 500
UPDATE_INTERVAL_MS = 50

class EEGViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG 16 canales - PyQtGraph")
        self.setFixedSize(1200, 800)

        # Layout principal
        layout = QtWidgets.QHBoxLayout(self)

        # Panel izquierdo (nombres y valores)
        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setAlignment(QtCore.Qt.AlignTop)
        self.labels = []

        for i in range(NUM_CHANNELS):
            row = QtWidgets.QHBoxLayout()

            name_label = QtWidgets.QLabel(f"Canal {i+1}")
            name_label.setFixedWidth(70)
            name_label.setFixedHeight(40)

            value_label = QtWidgets.QLabel("0.00")
            value_label.setFixedWidth(60)
            value_label.setAlignment(QtCore.Qt.AlignCenter)
            value_label.setStyleSheet(
                "background-color: white; border: 1px solid black; padding: 2px;"
            )
            value_label.setFixedHeight(40)

            row.addWidget(name_label)
            row.addWidget(value_label)
            left_panel.addLayout(row)
            self.labels.append(value_label)

        # Panel derecho (gráfica)
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.ci.layout.setColumnStretchFactor(0, 1)  # <-- clave
        self.plots = []
        self.curves = []

        for i in range(NUM_CHANNELS):
            p = self.plot_widget.addPlot(row=i, col=0)
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=False, y=False)
            p.hideAxis("left")
            p.setYRange(-2, 2)
            p.hideAxis("bottom")
            p.setFixedHeight(40)

            curve = p.plot(pen=pg.mkPen(color=(0, 255, 0), width=1))
            self.plots.append(p)
            self.curves.append(curve)

        # ----- eje de tiempo en fila extra -----
        axis_plot = self.plot_widget.addPlot(row=NUM_CHANNELS, col=0)
        axis_plot.hideAxis("left")
        axis_plot.showAxis("bottom")
        axis_plot.setLabel("bottom", "Tiempo (s)")
        axis_plot.getAxis("bottom").setTicks([[(t, f"{t:.0f}") for t in np.linspace(-10, 0, 10)]])
        axis_plot.setXRange(-10, 0)
        axis_plot.getViewBox().setDefaultPadding(0)
        axis_plot.getViewBox().setMouseEnabled(False, False)
        axis_plot.getViewBox().setYRange(0, 1)
        axis_plot.hideButtons()

        layout.addLayout(left_panel)
        layout.addWidget(self.plot_widget)

        # Datos iniciales
        self.data = np.zeros((NUM_CHANNELS, SAMPLES_PER_CHANNEL))

        # Timer para actualización
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(UPDATE_INTERVAL_MS)

    def update_data(self):
        # Genera datos aleatorios y desplaza
        new_values = np.random.normal(0, 0.3, size=(NUM_CHANNELS, 1))
        self.data = np.hstack((self.data[:, 1:], new_values))

        # Actualiza curvas y labels
        for i in range(NUM_CHANNELS):
            self.curves[i].setData(self.data[i])
            self.labels[i].setText(f"{self.data[i, -1]:.2f}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = EEGViewer()
    viewer.resize(1000, 600)
    viewer.show()
    sys.exit(app.exec_())