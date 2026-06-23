from PySide6.QtCore import Qt, QRectF, QLineF
from PySide6.QtWidgets import QApplication, QTableWidgetItem, QMainWindow, QGraphicsView, QGraphicsScene
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
from PySide6.QtGui import QPainter, QImage
import numpy as np
import sys
import os

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
                                         overwrite_contexts = DEFAULT_CONTEXT, reload=True, data_augmentation = True)

        
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
        self.ui.label.display(int(self.trajectory_label*100))


    # def paintEvent(self, event):
    #     pass
    #     # painter = QPainter(self)
    #     # for img, uiElem in self.imagesToShow:
    #     #     pSrcImage = uiElem.geometry()
    #     #     if len(img.shape)==3 and img.shape[2]==3:
    #     #         image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     #     else:
    #     #         image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #     #     image = cv2.resize(image, ((pSrcImage.width()//4)*4, pSrcImage.height()))
    #     #     image = QImage(image, (pSrcImage.width()//4)*4, pSrcImage.height(), QImage.Format_RGB888)
    #     #     painter.drawImage(pSrcImage.x(), pSrcImage.y(), image)
    #     # painter.end()

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
        features = self.traj_graphs[f][self.entity].x.view(-1).tolist()
        self.update_metrics_table(features)

        self.view = MyView(self.traj_graphs[f])
        self.view.setParent(self.ui.widget)
        self.view.show()
        self.ui.widget.setFixedSize(self.view.width(), self.view.height())

        self.update()


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
        rx, ry = self.graph['robot'].x[0,0].item()*cvtFactor, self.graph['robot'].x[0,1].item()*cvtFactor
        w, l = self.graph['robot'].x[0,4].item()*cvtFactor, self.graph['robot'].x[0,5].item()*cvtFactor
        item = self.scene.addEllipse(rx - w/2, ry - l/2, w, l, brush=Qt.yellow)
        self.nodeItems['robot'] = [(item, (rx, ry))]
        s, c = self.graph['robot'].x[0,2].item(), self.graph['robot'].x[0,3].item()
        self.scene.addLine(QLineF(rx, ry, rx+w*c, ry+w*s))#, pen=Qt.black)

        self.nodeItems['wall'] = []
        for w in range(self.graph['wall'].x.shape[0]):
            wx, wy = self.graph['wall'].x[w,0]*cvtFactor, self.graph['wall'].x[w,1]*cvtFactor
            r = 20
            item = self.scene.addEllipse(wx - r/2, wy - r/2, r, r, brush=Qt.red)
            self.nodeItems['wall'].append((item, (wx, wy)))

        self.nodeItems['human'] = []
        for h in range(self.graph['human'].x.shape[0]):
            hx, hy = self.graph['human'].x[h,0]*cvtFactor, self.graph['human'].x[h,1]*cvtFactor
            r = 30
            item = self.scene.addEllipse(hx - r/2, hy - r/2, r, r, brush=Qt.blue)
            self.nodeItems['human'].append((item, (hx, hy)))
            s, c = self.graph['human'].x[h,2].item(), self.graph['human'].x[h,3].item()
            self.scene.addLine(QLineF(hx, hy, hx+r*c, hy+r*s))#, pen=Qt.black)

        self.nodeItems['object'] = []
        for o in range(self.graph['object'].x.shape[0]):
            ox, oy = self.graph['object'].x[o,0]*cvtFactor, self.graph['object'].x[o,1]*cvtFactor
            w, l = self.graph['object'].x[o,4].item()*cvtFactor, self.graph['object'].x[o,5].item()*cvtFactor
            item = self.scene.addEllipse(ox - w/2, oy - l/2, w, l, brush=Qt.magenta)
            self.nodeItems['object'].append((item,(ox,oy)))
            s, c = self.graph['object'].x[o,2].item(), self.graph['object'].x[o,3].item()
            self.scene.addLine(QLineF(ox, oy, ox+w*c, oy+w*s))#, pen=Qt.black)


        gx, gy = self.graph['goal'].x[0,0].item()*cvtFactor, self.graph['goal'].x[0,1].item()*cvtFactor
        w = self.graph['goal'].x[0,4].item()*cvtFactor
        item = self.scene.addEllipse(gx - w/2, gy - w/2, w, w, brush=Qt.green)
        self.nodeItems['goal'] = [(item, (gx,gy))]
        s, c = self.graph['goal'].x[0,2].item(), self.graph['goal'].x[0,3].item()
        self.scene.addLine(QLineF(gx, gy, gx+w*c, gy+w*s))#, pen=Qt.black)

        sx, sy = -400, -400
        r = 40
        item = self.scene.addEllipse(sx - r/2, sy - r/2, r, r, brush=Qt.cyan)
        self.nodeItems['scenario'] = [(item, (sx,sy))]

        for etype, edges in self.graph.edge_index_dict.items():
            src, rel, dst = etype[0], etype[1], etype[2]
            if rel == 'self' or src == 'scenario' or dst == 'scenario':
                continue
            # print(edges)
            for e in range(edges.shape[1]):
                x1, y1 = self.nodeItems[src][edges[0,e]][1][0], self.nodeItems[src][edges[0,e]][1][1]
                x2, y2 = self.nodeItems[dst][edges[1,e]][1][0], self.nodeItems[dst][edges[1,e]][1][1]
                self.scene.addLine(QLineF(x1, y1, x2, y2))

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
