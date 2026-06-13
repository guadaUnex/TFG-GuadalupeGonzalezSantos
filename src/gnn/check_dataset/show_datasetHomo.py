from PySide6.QtCore import Qt, QRectF, QLineF
from PySide6.QtWidgets import QApplication, QTableWidgetItem, QMainWindow, QGraphicsView, QGraphicsScene
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile
from PySide6.QtGui import QPainter, QImage
import numpy as np
import sys
import os
import copy
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'..'))

from datasetHomo import SocNavHomoDataset, collate

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

        self.data_sequence = SocNavHomoDataset(trajectory_file, data_path = dataroot, context_path = contextQ_file,
                                         overwrite_contexts = DEFAULT_CONTEXT, data_augmentation = True)
        
        self.all_features = self.data_sequence.all_features

        
        self.ui.trajectory_scrollbar.setMaximum(len(self.data_sequence)-1)
        self.ui.trajectory_scrollbar.valueChanged.connect(self.load_trajectory)

        self.load_trajectory(0)

        self.ui.show()
        self.resize(self.ui.geometry().width(), self.ui.geometry().height())

    def set_entity_table(self, entity):
        entity_len = 0
        self.entity = entity
        for frame in self.traj_graphs:
            if frame['x'][self.entity].shape[0]>entity_len:
                entity_len = frame['x'][self.entity].shape[0]

        self.ini_metrics_table(self.features[self.entity], entity_len)

        self.show_frame(self.ui.frame_scrollbar.value())

    def fromHomoToHeteroGraphSeq(self, homoGraphs, metrics):
        heteroGraphs = []
        print(len(homoGraphs))
        print(len(metrics))
        metrics_list = list(torch.unbind(metrics, dim=0))
        for graph, metricsF in zip(homoGraphs, metrics_list):
            newGraph = {}
            newGraph['x'] = {}
            newGraph['x']['scenario'] = metricsF.reshape(1, -1)
            rel_idx = []
            for entity in self.entities:
                if entity!='scenario':
                    idx = self.all_features.index(entity)
                    feats = graph.x[graph.x[:,idx]==1.,:]
                    # print(graph.x)
                    feats = feats[:,len(self.entities)-1:]
                    newGraph['x'][entity] = feats
                    for i in range(feats.shape[0]):
                        rel_idx.append((entity, i))
            newGraph['edge_index'] = graph.edge_index
            newGraph['rel_idx'] = rel_idx
            heteroGraphs.append(newGraph)    
        return heteroGraphs

    def load_trajectory(self, idx):
        self.entities = copy.copy(self.data_sequence.type_features)
        self.all_features = self.data_sequence.all_features
        self.features = {}
        for entity in self.entities:
            self.features[entity] = self.data_sequence.geometric_features
        self.features['scenario'] = self.data_sequence.metrics_features +  self.data_sequence.context_features
        self.entities += ['scenario']
        seq, metrics, label, seq_len = self.data_sequence[idx]
        self.traj_graphs = self.fromHomoToHeteroGraphSeq(seq, metrics)
        self.trajectory_label = label[0].item() 
        traj_steps = seq_len.item()
        self.ui.frame_scrollbar.setMaximum(traj_steps-1)
        self.entity = self.ui.entityKey.currentText()
        self.set_entity_table(self.entity)
        self.ui.label.display(int(self.trajectory_label*100))


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
        features = self.traj_graphs[f]['x'][self.entity].reshape(-1).tolist()
        self.update_metrics_table(features)

        self.view = MyView(self.traj_graphs[f], self.data_sequence.geometric_features)
        self.view.setParent(self.ui.widget)
        self.view.show()
        self.ui.widget.setFixedSize(self.view.width(), self.view.height())

        self.update()


class MyView(QGraphicsView):
    def __init__(self, graph, geometric_features):
        super().__init__()
        self.graph = graph
        self.geometric_features = geometric_features
        self.scene = QGraphicsScene(self)
        self.nodeItems = dict()
        self.setFixedSize(1002, 1002)
        self.create_scene()
    
    def create_scene(self):
        x_idx = self.geometric_features.index('x')
        y_idx = self.geometric_features.index('y')
        s_idx = self.geometric_features.index('sin_a')
        c_idx = self.geometric_features.index('cos_a')
        w_idx = self.geometric_features.index('w')
        l_idx = self.geometric_features.index('l')
        th_idx = self.geometric_features.index('th_pos')
        cvtFactor = 500
        self.scene.setSceneRect(QRectF(-500, -500, 1000, 1000))
        rx, ry = self.graph['x']['robot'][0,x_idx].item()*cvtFactor, self.graph['x']['robot'][0,y_idx].item()*cvtFactor
        w, l = self.graph['x']['robot'][0,w_idx].item()*cvtFactor, self.graph['x']['robot'][0,l_idx].item()*cvtFactor
        item = self.scene.addEllipse(rx - w/2, ry - l/2, w, l, brush=Qt.yellow)
        self.nodeItems['robot'] = [(item, (rx, ry))]
        s, c = self.graph['x']['robot'][0,s_idx].item(), self.graph['x']['robot'][0,c_idx].item()
        self.scene.addLine(QLineF(rx, ry, rx+w*c, ry+w*s))#, pen=Qt.black)

        self.nodeItems['wall'] = []
        for w in range(self.graph['x']['wall'].shape[0]):
            wx, wy = self.graph['x']['wall'][w,x_idx]*cvtFactor, self.graph['x']['wall'][w,y_idx]*cvtFactor
            r = 20
            item = self.scene.addEllipse(wx - r/2, wy - r/2, r, r, brush=Qt.red)
            self.nodeItems['wall'].append((item, (wx, wy)))

        self.nodeItems['human'] = []
        for h in range(self.graph['x']['human'].shape[0]):
            hx, hy = self.graph['x']['human'][h,x_idx]*cvtFactor, self.graph['x']['human'][h,y_idx]*cvtFactor
            r = 30
            item = self.scene.addEllipse(hx - r/2, hy - r/2, r, r, brush=Qt.blue)
            self.nodeItems['human'].append((item, (hx, hy)))
            s, c = self.graph['x']['human'][h,s_idx].item(), self.graph['x']['human'][h,c_idx].item()
            self.scene.addLine(QLineF(hx, hy, hx+r*c, hy+r*s))#, pen=Qt.black)

        self.nodeItems['object'] = []
        for o in range(self.graph['x']['object'].shape[0]):
            ox, oy = self.graph['x']['object'][o,x_idx]*cvtFactor, self.graph['x']['object'][o,y_idx]*cvtFactor
            w, l = self.graph['object'].x[o,w_idx].item()*cvtFactor, self.graph['x']['object'][o,l_idx].item()*cvtFactor
            item = self.scene.addEllipse(ox - w/2, oy - l/2, w, l, brush=Qt.magenta)
            self.nodeItems['object'].append((item,(ox,oy)))
            s, c = self.graph['x']['object'][o,s_idx].item(), self.graph['x']['object'][o,c_idx].item()
            self.scene.addLine(QLineF(ox, oy, ox+w*c, oy+w*s))#, pen=Qt.black)


        gx, gy = self.graph['x']['goal'][0,x_idx].item()*cvtFactor, self.graph['x']['goal'][0,y_idx].item()*cvtFactor
        w = self.graph['x']['goal'][0,th_idx].item()*cvtFactor
        item = self.scene.addEllipse(gx - w/2, gy - w/2, w, w, brush=Qt.green)
        self.nodeItems['goal'] = [(item, (gx,gy))]
        s, c = self.graph['x']['goal'][0,s_idx].item(), self.graph['x']['goal'][0,c_idx].item()
        self.scene.addLine(QLineF(gx, gy, gx+w*c, gy+w*s))#, pen=Qt.black)

        n_edges = self.graph['edge_index'].shape[1]
        for e in range(n_edges):
            src=self.graph['edge_index'][0,e]
            dst=self.graph['edge_index'][1,e]
            if src!=dst:
                src_ent, src_idx = self.graph['rel_idx'][src]
                dst_ent, dst_idx = self.graph['rel_idx'][dst]
                x1, y1 = self.nodeItems[src_ent][src_idx][1][0], self.nodeItems[src_ent][src_idx][1][1]
                x2, y2 = self.nodeItems[dst_ent][dst_idx][1][0], self.nodeItems[dst_ent][dst_idx][1][1]
                self.scene.addLine(QLineF(x1, y1, x2, y2))

        # for etype, edges in self.graph.edge_index_dict.items():
        #     src, rel, dst = etype[0], etype[1], etype[2]
        #     if rel == 'self' or src == 'scenario' or dst == 'scenario':
        #         continue
        #     # print(edges)
        #     for e in range(edges.shape[1]):
        #         x1, y1 = self.nodeItems[src][edges[0,e]][1][0], self.nodeItems[src][edges[0,e]][1][1]
        #         x2, y2 = self.nodeItems[dst][edges[1,e]][1][0], self.nodeItems[dst][edges[1,e]][1][1]
        #         self.scene.addLine(QLineF(x1, y1, x2, y2))

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
