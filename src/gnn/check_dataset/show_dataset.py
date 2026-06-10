from PySide6.QtCore import Qt, QRectF
from PySide6.QtWidgets import QApplication, QTableWidgetItem, QMainWindow, QGraphicsView, QGraphicsScene
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
from PySide6.QtGui import QPainter, QImage
import cv2
import numpy as np
import copy
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..'))

from dataset import SocNavHeteroDataset, collate

DEFAULT_CONTEXT = "A robot is working with lab samples. The samples contain a deadly virus."

IMG_WIDTH = 1000
IMG_HEIGHT = 1000

class MainWindow(QMainWindow):
    def __init__(self, trajectory_file, contextQ_file, dataroot):
        super().__init__()

        ui_path = os.path.join(os.path.dirname(__file__),"dataset_viewer.ui")
        ui_file = QFile(ui_path)
        ui_file.open(QFile.ReadOnly)

        loader = QUiLoader()
        self.ui = loader.load(ui_file, self)
        ui_file.close()

        self.ui.frame_scrollbar.valueChanged.connect(self.show_frame)
        self.ui.entityKey.currentTextChanged.connect(self.set_entity_table)

        self.contextQ_file = contextQ_file

        self.data_sequence = SocNavHeteroDataset(trajectory_file, data_path = dataroot, context_path = contextQ_file,
                                         overwrite_contexts = DEFAULT_CONTEXT, data_augmentation = True)

        
        self.ui.trajectory_scrollbar.setMaximum(len(self.data_sequence)-1)
        self.ui.trajectory_scrollbar.valueChanged.connect(self.load_trajectory)

        self.load_trajectory(0)

        self.ui.show()
        self.resize(self.ui.geometry().width(), self.ui.geometry().height())

    def set_entity_table(self, entity):
        entity_len = 0
        self.entity = entity
        for frame in self.traj_graphs:
            if frame[self.entity].x.shape[0]>entity_len:
                entity_len = frame[self.entity].x.shape[0]

        self.ini_metrics_table(self.features[self.entity], entity_len)

        self.show_frame(self.ui.frame_scrollbar.value())

    def load_trajectory(self, idx):

        self.features = self.data_sequence.all_feature_names
        seq, label, seq_len = self.data_sequence[idx]
        self.traj_graphs = seq 
        self.trajectory_label = label[0].item() 
        traj_steps = seq_len.item()
        self.ui.frame_scrollbar.setMaximum(traj_steps-1)
        self.entity = self.ui.entityKey.currentText()
        self.set_entity_table(self.entity)
        # entity_len = 0
        # for frame in self.traj_graphs:
        #     if frame[self.entity].x.shape(0)>entity_len:
        #         entity_len = frame[self.entity].x.shape(0)

        # self.ini_metrics_table(self.features[self.entity], entity_len)

        # self.show_frame(self.ui.frame_scrollbar.value())


        # tensor_trajectory = self.data_sequence.current_trajectory
        # self.trajectory = tensor_to_sequence(tensor_trajectory) #self.data_sequence.orig_data[idx]
        # self.ini_metrics_table()

        # traj_steps = len(self.trajectory['sequence'])
        # self.ui.frame_scrollbar.setMaximum(traj_steps-1)

        # x_min, y_min, x_max, y_max = self.compute_scenario_limits(self.trajectory)

        # FR = self.compute_scenario_FR(x_min, y_min, x_max, y_max)

        # self.scenario_img, self.GRID_CELL_SIZEX, self.GRID_CELL_SIZEY, self.GRID_X_ORIG, self.GRID_Y_ORIG, self.GRID_ANGLE_ORIG, self.GRID_HEIGHT = draw_scenario(self.trajectory, IMG_WIDTH, IMG_HEIGHT, FR)

        # self.frame_img = copy.deepcopy(self.scenario_img)

        # self.imagesToShow = [(self.frame_img, self.ui.scenario_frame)]

        # self.human_colors = dict()

        # self.show_frame(self.ui.frame_scrollbar.value())


    def paintEvent(self, event):
        pass
        # painter = QPainter(self)
        # for img, uiElem in self.imagesToShow:
        #     pSrcImage = uiElem.geometry()
        #     if len(img.shape)==3 and img.shape[2]==3:
        #         image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     else:
        #         image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #     image = cv2.resize(image, ((pSrcImage.width()//4)*4, pSrcImage.height()))
        #     image = QImage(image, (pSrcImage.width()//4)*4, pSrcImage.height(), QImage.Format_RGB888)
        #     painter.drawImage(pSrcImage.x(), pSrcImage.y(), image)
        # painter.end()

    def ini_metrics_table(self, features, max_number):

        self.ui.metrics_table.setRowCount(len(features)*max_number)
        idx = 0
        for n in range(max_number):
            for ft in features:
                self.ui.metrics_table.setItem(idx, 0, QTableWidgetItem(ft+'_'+str(n)))
                self.ui.metrics_table.setItem(idx, 1, QTableWidgetItem("{:4f}".format(0)))
                idx+=1

        self.ui.metrics_table.resizeColumnsToContents()            

    def update_metrics_table(self, ft_list):
        for idx, feature in enumerate(ft_list):
            self.ui.metrics_table.setItem(idx, 1, QTableWidgetItem("{:4f}".format(feature)))
        self.ui.metrics_table.resizeColumnsToContents()            


    def show_frame(self, f):
        # self.frame_img = copy.deepcopy(self.scenario_img)

        # self.frame_img, self.human_colors = draw_frame(self.trajectory['sequence'][f], self.frame_img, self.human_colors, self.GRID_CELL_SIZEX, 
        #                                       self.GRID_CELL_SIZEY, self.GRID_X_ORIG, self.GRID_Y_ORIG, 
        #                                       self.GRID_ANGLE_ORIG, self.GRID_HEIGHT)

        # self.imagesToShow = [(self.frame_img, self.ui.scenario_frame)]
        features = self.traj_graphs[f][self.entity].x.view(-1).tolist()
        self.update_metrics_table(features)

        self.view = MyView(self.traj_graphs[f])
        self.view.setParent(self.ui.widget)
        self.view.show()
        self.ui.widget.setFixedSize(self.view.width(), self.view.height())


        self.update()

    def compute_scenario_limits(self, trajectory):
        x_min, y_min = np.inf, np.inf
        x_max, y_max = -np.inf, -np.inf

        for w in trajectory['walls']:
            x_min = np.minimum(np.minimum(x_min, w[0]), w[2])
            x_max = np.maximum(np.maximum(x_max, w[0]), w[2])
            y_min = np.minimum(np.minimum(y_min, w[1]), w[3])
            y_max = np.maximum(np.maximum(y_max, w[1]), w[3])
        for s in trajectory['sequence']:
            for o in s['objects']:
                x_min = np.minimum(x_min, o['x'])
                x_max = np.maximum(x_max, o['x'])
                y_min = np.minimum(y_min, o['y'])
                y_max = np.maximum(y_max, o['y'])
            for p in s['people']:
                x_min = np.minimum(x_min, p['x'])
                x_max = np.maximum(x_max, p['x'])
                y_min = np.minimum(y_min, p['y'])
                y_max = np.maximum(y_max, p['y'])

            x_min = np.minimum(x_min, s['robot']['x'])
            x_max = np.maximum(x_max, s['robot']['x'])
            y_min = np.minimum(y_min, s['robot']['y'])
            y_max = np.maximum(y_max, s['robot']['y'])

            x_min = np.minimum(x_min, s['goal']['x'])
            x_max = np.maximum(x_max, s['goal']['x'])
            y_min = np.minimum(y_min, s['goal']['y'])
            y_max = np.maximum(y_max, s['goal']['y'])
        return x_min-2.5, y_min-2.5, x_max+2.5, y_max+2.5
    
    def compute_scenario_FR(self, x_min, y_min, x_max, y_max):
        FR = dict()
        H = y_max-y_min
        W = x_max-x_min
        FR["height"] = IMG_HEIGHT
        FR["width"] = IMG_WIDTH
        FR["cell_size"] = np.maximum(H / IMG_HEIGHT, W / IMG_WIDTH) 
        FR["x_orig"] = x_min
        FR["y_orig"] = y_min 
        FR["angle_orig"] = 0

        print(H, W, x_min, y_min, x_max, y_max, FR)

        return FR

class MyView(QGraphicsView):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.scene = QGraphicsScene(self)
        self.nodeItems = dict()
        self.setFixedSize(1002, 1002)
        self.create_scene()
    
    def create_scene(self):
        cvtFactor = 500
        self.scene.setSceneRect(QRectF(-500, -500, 1000, 1000))
        rx, ry = self.graph['robot'].x[0,0]*cvtFactor, self.graph['robot'].x[0,1]*cvtFactor
        w, l = self.graph['robot'].x[0,4]*cvtFactor, self.graph['robot'].x[0,5]*cvtFactor
        item = self.scene.addEllipse(rx - w/2, ry - l/2, w, l, brush=Qt.darkRed)
        self.nodeItems['robot'] = item

        self.setScene(self.scene)

if __name__ == "__main__":
    if len(sys.argv)<3:
        print("Specify a json/txt file containing the trajectories,a context quantization file, and the path to the data")
        exit()

    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts)

    app = QApplication(sys.argv)

    window = MainWindow(sys.argv[1], sys.argv[2], sys.argv[3])

    window.show()  

    sys.exit(app.exec())
