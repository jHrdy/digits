import sys
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, QRect
from model import Net 
import torch.nn.functional as F

CELL_SIZE = 35  
GRID_SIZE = 32  

class DrawGrid(QWidget):
    def __init__(self):
        super().__init__()
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.drawing = False
        self.initUI()
        self.tensor_result = None

    def initUI(self):
        self.setWindowTitle("Draw MNIST-like digit")
        self.setGeometry(100, 100, CELL_SIZE*GRID_SIZE, CELL_SIZE*GRID_SIZE + 50)

        self.submit_btn = QPushButton("SUBMIT", self)
        self.submit_btn.clicked.connect(self.submit)
        self.submit_btn.setGeometry(10, CELL_SIZE*GRID_SIZE + 10, 100, 30)
        self.clear_btn = QPushButton("CLEAR", self)
        self.clear_btn.clicked.connect(self.clear_grid)
        self.clear_btn.setGeometry(120, CELL_SIZE*GRID_SIZE + 10, 100, 30)

        self.show()
    
    def clear_grid(self):
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.update()  # prekreslí widget

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                color = QColor(255, 255, 255) if self.grid[i][j] == 0 else QColor(0,0,0)
                qp.fillRect(QRect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE), color)
                qp.setPen(Qt.gray)
                qp.drawRect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        qp.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.draw_cell(event)

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.draw_cell(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def draw_cell(self, event):
        x = event.pos().x() // CELL_SIZE
        y = event.pos().y() // CELL_SIZE
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            self.grid[y][x] = 1
            self.update()

    def submit(self):
        
        self.tensor_result = torch.tensor(self.grid, dtype=torch.float32)
        #print("Tensor shape:", self.tensor_result.shape)
        #print(self.tensor_result)

        net = Net()  
        net.load_state_dict(torch.load("mnist_model_weights.pth"))
        net.eval()
        x = self.tensor_result.unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=(28,28), mode='bilinear', align_corners=False)
        x = x.view(1, -1)
        
        print(f'Neural network label: {net(x).argmax()}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DrawGrid()
    sys.exit(app.exec_())