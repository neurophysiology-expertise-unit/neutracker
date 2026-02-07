from .tracker import MPTracker,ellipseToContour
import numpy as np
import cv2
import os
from PyQt5.QtWidgets import (QWidget,
                             QApplication,
                             QVBoxLayout,
                             QGridLayout,
                             QFormLayout,
                             QCheckBox,
                             QComboBox,
                             QTextEdit,
                             QSlider,
                             QPushButton,
                             QLabel,
                             QGraphicsView,
                             QGraphicsScene,
                             QGraphicsItem,
                             QGraphicsLineItem,
                             QGroupBox,
                             QTableWidget,
                             QTabWidget,
                             QFileDialog)
from PyQt5.QtGui import QImage, QPixmap,QBrush,QPen,QColor,QFont
from PyQt5.QtCore import Qt,QSize,QRectF,QLineF,QPointF

useOpenGL=False
class MptrackerDisplay(QWidget):
    def __init__(self,img = None,runFunct = None):
        super(MptrackerDisplay,self).__init__()
        grid = QFormLayout()
        
        self.wNFrames = QLabel('')
        self.wNFrames.setMaximumHeight(25)
        self.wNFrames.setMaximumWidth(200)
        grid.addRow(QLabel('Frame number:'),self.wNFrames)
        if not runFunct is None:
            runbutton = QPushButton('Run')
            runbutton.clicked.connect(runFunct)
            self.wParallel = QCheckBox('Parallel')
            grid.addRow(runbutton, self.wParallel)
        self.wFrame = QSlider(Qt.Horizontal)
        self.wFrame.valueChanged.connect(self.processFrame)
        self.wFrame.setMaximum(1)
        self.wFrame.setMinimum(0)
        self.wFrame.setValue(0)
        if not runFunct is None:
            grid.addRow(runbutton,self.wFrame)
        else:
            grid.addRow(self.wFrame)
        # images and plots
        self._init_pg()
        self.setImage(img)
        grid.addRow(self.win)
        self.setLayout(grid)

    def _init_pg(self):
        import pyqtgraph as pg
        pg.setConfigOption('background', [200,200,200])
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.win = pg.GraphicsLayoutWidget()
        p1 = self.win.addPlot(title="")
        p1.getViewBox().invertY(True)
        #p1.hideAxis('left')
        #p1.hideAxis('bottom')
        self.imgview = pg.ImageItem(useOpenGL=useOpenGL)
        self.imgview.setScale(1)
        self.imgview.setPos(0,0)
        #self.pltPup = pg.PlotCurveItem([np.nan,np.nan],
        #                               [np.nan,np.nan])
        
        #p1.addItem(self.pltPup)
        #self.text = pg.TextItem('hello',
        #                        color = [200,100,100],
        #                        anchor = [1,0])
        #b=QFont()
        #b.setPixelSize(24)
        #self.text.setFont(b)
        #p1.addItem(self.text)
        self.roi_selection = EyeROIWidget()
        elements = [self.imgview] + self.roi_selection.items()
        [p1.addItem(e) for e in elements]
        self.p1 = p1

    def processFrame(self):
        pass
    def setImage(self,image):
        self.imgview.setImage(image)
    def setPupilOutline(self,pos,radius):
        if np.isnan(radius):
            return
        c = ellipseToContour(pos,radius,radius,0)
        self.pltPup.setData(x = c[:,0,1],y = c[:,0,0])

class EyeROIWidget():
    def __init__(self,locations = None):
        import pyqtgraph as pg
        if not locations is None:
            p1 = locations[0]
            p2 = locations[2]
            p3 = locations[1]
            p4 = locations[3]
        else:
            p1,p2 = ([100,200],[200,200])
            p3,p4 = ([150,250],[150,150])
            self.roi_corners = pg.PolyLineROI((p1,p3,p2,p4),pen = (1,9))
        self.roi_points_selected = None
        def updateroi(val):
            p = self.roi_corners.pos()
            c = self.roi_corners.getHandles()
            self.roi_points_selected = np.stack([
                np.array(v.pos()+p).astype(int) for v in c])
        self.roi_corners.sigRegionChanged.connect(updateroi)
    def items(self):
        return [self.roi_corners]
    def get(self):
        return self.roi_points_selected
    def set(self,points):
        p1 = points[0]
        p2 = points[2]
        p3 = points[1]
        p4 = points[3]
        self.roi_corners.setPos(0,0)
        c = self.roi_corners.getHandles()# + self.roi_lid.getHandles()
        c[0].setPos(*p1)
        c[1].setPos(*p2)
        c[2].setPos(*p3)
        c[3].setPos(*p4)
    def setVisible(self,value):
        self.roi_corners.setVisible(value)
        
class MptrackerParameters(QWidget):
    def __init__(self,tracker,
                 image = np.ones((75,75),
                                 dtype=np.uint8),
                 eyewidget = None,
):
        super(MptrackerParameters,self).__init__()
        self.tracker = tracker
        self.parameters = tracker.parameters
        # Set the layout and the tabs widget
        #     - Tab "Tracker parameters"
        #     - Tab "ROI selection"
        #     - Tab "Display and save"
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabTrackParameters = QWidget()
        self.tabROI = QWidget()
        self.tabDisplay = QWidget()
        layout.addWidget(self.tabs)
        self.tabs.addTab(self.tabTrackParameters,'Tracker parameters')
        #self.tabs.addTab(self.tabROI,'Tracker ROI selection')
        self.tabs.addTab(self.tabDisplay,'Display and save')

        # Image parameters
        self.tabTrackParameters.layout = QGridLayout()
        pGroup = QGroupBox(self)
        pGrid = QFormLayout()
        pGroup.setLayout(pGrid)
        pGroup.setTitle("Image enhancement")  
        self.tabTrackParameters.setLayout(self.tabTrackParameters.layout)

        # Gamma
        self.wGamma = QSlider(Qt.Horizontal)
        self.wGamma.setValue(int(self.parameters['gamma']*10))
        self.wGamma.setMaximum(30)
        self.wGamma.setMinimum(1)
        self.wGamma.setSingleStep(5)
        self.wGammaLabel = QLabel('Gamma [{0}]:'.format(
            self.wGamma.value()/10.))
        self.wGamma.valueChanged.connect(self.setGamma)
        pGrid.addRow(self.wGammaLabel, self.wGamma)
        
        # blur filter
        self.wFilterType = QComboBox()
        [self.wFilterType.addItem(v) for v in ['median','gaussian']]
        def update_type(val):
            self.tracker.parameters['filterType'] = val
            self.update()
        self.wFilterType.currentTextChanged.connect(update_type)
        pGrid.addRow(QLabel('Filter type'), self.wFilterType)
        # blur size
        self.wFilterSize = QSlider(Qt.Horizontal)
        self.wFilterSize.setValue(int(self.parameters['filterSize']*10))
        self.wFilterSize.setMaximum(61)
        self.wFilterSize.setMinimum(1)
        self.wFilterSize.setSingleStep(2)
        self.wFilterSizeLabel = QLabel('Filter size [{0}]:'.format(
            self.wFilterSize.value()/10.))
        def update_size(value):
            # make sure value is odd
            if not np.mod(value,2) == 1:
                value += 1
            self.parameters['filterSize'] = int(value)
            self.wFilterSizeLabel.setText('filter size [{0}]:'.format(int(value)))
            self.update()
        self.wFilterSize.valueChanged.connect(update_size)
        pGrid.addRow(self.wFilterSizeLabel, self.wFilterSize)

        # Contrast equalization (adaptative or full)
        self.wContrastLim = QSlider(Qt.Horizontal)
        self.wContrastLim.setValue(int(self.parameters['contrast_clipLimit']))
        self.wContrastLim.setMinimum(0)
        self.wContrastLim.setMaximum(200)
        self.wContrastLimLabel = QLabel('Contrast limit [{0}]:'.format(
            self.wContrastLim.value()))
        self.wContrastLim.valueChanged.connect(self.setContrastLim)
        pGrid.addRow(self.wContrastLimLabel,self.wContrastLim)

        self.wContrastGridSize = QSlider(Qt.Horizontal)
        self.wContrastGridSize.setValue(int(self.parameters['contrast_gridSize']))
        self.wContrastGridSize.setMaximum(200)
        self.wContrastGridSize.setMinimum(1)
        self.wContrastGridSize.setSingleStep(1)
        self.wContrastGridSizeLabel = QLabel('Contrast grid size [{0}]:'.format(
            self.wContrastGridSize.value()))
        self.wContrastGridSize.valueChanged.connect(self.setContrastGridSize)
        pGrid.addRow(self.wContrastGridSizeLabel,self.wContrastGridSize)
        
        # Morphological operations
        pGroup2 = QGroupBox()
        pGrid2 = QFormLayout()
        pGroup2.setLayout(pGrid2)
        pGroup2.setTitle("Morphological operations")  

        self.wOpenKernelSize = QSlider(Qt.Horizontal)
        self.wOpenKernelSize.setValue(int(self.parameters['open_kernelSize']))
        self.wOpenKernelSize.setMaximum(61)
        self.wOpenKernelSize.setMinimum(0)
        self.wOpenKernelSize.setSingleStep(1)
        self.wOpenKernelSizeLabel = QLabel('Morph open size [{0}]:'.format(
            self.wOpenKernelSize.value()))
        self.wOpenKernelSize.valueChanged.connect(self.setOpenKernelSize)
        pGrid2.addRow(self.wOpenKernelSizeLabel, self.wOpenKernelSize)

        self.wCloseKernelSize = QSlider(Qt.Horizontal)
        self.wCloseKernelSize.setValue(int(self.parameters['close_kernelSize']))
        self.wCloseKernelSize.setMaximum(61)
        self.wCloseKernelSize.setMinimum(0)
        self.wCloseKernelSize.setSingleStep(1)
        self.wCloseKernelSizeLabel = QLabel('Morph close size [{0}]:'.format(
            self.wCloseKernelSize.value()))
        self.wCloseKernelSize.valueChanged.connect(self.setCloseKernelSize)
        pGrid2.addRow(self.wCloseKernelSizeLabel, self.wCloseKernelSize)

        pGroup3 = QGroupBox()
        pGrid3 = QFormLayout()
        pGroup3.setLayout(pGrid3)
        pGroup3.setTitle("Detection")  
        
        self.wBinThreshold = QSlider(Qt.Horizontal)
        self.wBinThreshold.setValue(int(self.parameters['threshold']))
        self.wBinThresholdLabel = QLabel('Binary contrast [40]:')
        self.wBinThreshold.setMinimum(0)
        self.wBinThreshold.setMaximum(255)
        self.wBinThreshold.valueChanged.connect(self.setBinThreshold)
        pGrid3.addRow(self.wBinThresholdLabel,self.wBinThreshold)

        self.wRoundIndex = QTextEdit(str(self.parameters['roundIndex']))
        self.wRoundIndex.setMaximumHeight(25)
        self.wRoundIndex.setMaximumWidth(60)
        self.wRoundIndex.textChanged.connect(self.setRoundIndex)
        pGrid3.addRow(QLabel('Circle threshold:'),self.wRoundIndex)

        self.wInvertThreshold = QCheckBox()
        self.wInvertThreshold.setChecked(self.parameters['invertThreshold'])
        self.wInvertThreshold.stateChanged.connect(self.setInvertThreshold)
        pGrid3.addRow(QLabel('White pupil:'),self.wInvertThreshold)

        self.wDisableCRtrack = QCheckBox()
        self.wDisableCRtrack.setChecked(self.parameters['crTrack'])
        self.wDisableCRtrack.stateChanged.connect(self.setCRTrack)
        pGrid3.addRow(QLabel('Track corneal reflection:'),self.wDisableCRtrack)

        self.wSequentialCrMode = QCheckBox()
        self.wSequentialCrMode.setChecked(self.parameters['sequentialCrMode'])
        self.wSequentialCrMode.stateChanged.connect(self.setSequentialCrMode)
        pGrid3.addRow(QLabel('Sequential refraction mode:'),self.wSequentialCrMode)

        self.wSequentialPupilMode = QCheckBox()
        self.wSequentialPupilMode.setChecked(self.parameters['sequentialPupilMode'])
        self.wSequentialPupilMode.stateChanged.connect(self.setSequentialPupilMode)
        pGrid3.addRow(QLabel('Sequential pupil mode:'),self.wSequentialPupilMode)

        self.wResetSequentialPupil = QPushButton('Reset sequential pupil')
        self.wResetSequentialPupil.clicked.connect(self.resetSequentialPupil)
        pGrid3.addRow(self.wResetSequentialPupil)

        pGroup4 = QGroupBox()
        pGrid4 = QFormLayout()
        pGroup4.setLayout(pGrid4)
        pGroup4.setTitle("ROI")  

        self.wPoints = QLabel('nan,nan \n nan,nan \n nan,nan \n nan,nan\n')
        pGrid4.addRow(QLabel('ROI points:'),self.wPoints)
        self.wPoints.mouseDoubleClickEvent = self.clearPoints
        def setpoints(points):
            self.wPoints.setText(
                ' \n'.join([','.join([str(w) for w in p]) for p in points]))
        if not eyewidget is None:
            if len(self.parameters['points']) == 4:
                eyewidget.set(self.parameters['points'])
                setpoints(self.parameters['points'])
            def updateROI(val):
                points = eyewidget.get()
                if not points is None:
                    p = []
                    for t in points:
                        x,y = t
                        p.append([x,y])
                    self.parameters['points'] = p
                    self.tracker.setROI(self.parameters['points'])
                    self.tracker.parameters['pupilApprox'] = None
                    setpoints(points)
                    self.update()
            button = QPushButton('Update ROI')
            button.clicked.connect(updateROI)
            pGrid4.addRow(button)
            self.wshowROI = QCheckBox()
            self.wshowROI.setChecked(True)
            def setshowROI(value):
                eyewidget.setVisible(value)
            self.wshowROI.stateChanged.connect(setshowROI)
            pGrid4.addRow(QLabel('Show ROI:'),self.wshowROI)


        # parameters, buttons and options        
        
        self.tabTrackParameters.layout.addWidget(pGroup)
        self.tabTrackParameters.layout.addWidget(pGroup2)
        self.tabTrackParameters.layout.addWidget(pGroup3)
        self.tabTrackParameters.layout.addWidget(pGroup4)

        self.tabDisplay.layout = QGridLayout()
        pGroup = QGroupBox()
        self.pGridSave = QFormLayout()
        pGroup.setLayout(self.pGridSave)
        pGroup.setTitle("Display parameters")
        self.tabDisplay.setLayout(self.tabDisplay.layout)
        
        self.wEyeRadius = QTextEdit(str(self.parameters['eye_radius_mm']))
        self.wEyeRadius.setMaximumHeight(25)
        self.wEyeRadius.setMaximumWidth(60)
        self.wEyeRadius.textChanged.connect(self.setEyeRadius)
        self.pGridSave.addRow(QLabel('Approximate eye radius (mm):'),self.wEyeRadius)

        # Field of View Width (mm)
        self.wFOV = QTextEdit(str(self.parameters.get('fov_mm', 0.0)))
        self.wFOV.setMaximumHeight(25)
        self.wFOV.setMaximumWidth(60)
        self.wFOV.textChanged.connect(self.setFOV)
        self.pGridSave.addRow(QLabel('Image Field of View width (mm):'),self.wFOV)




        self.wDisplayBinaryImage = QCheckBox()
        self.wDisplayBinaryImage.setChecked(False)
        self.wDisplayBinaryImage.stateChanged.connect(self.updateTrackerOutputBinaryImage)
        self.pGridSave.addRow(QLabel('Display binary image:'),self.wDisplayBinaryImage)

        self.wDrawProcessed = QCheckBox()
        self.wDrawProcessed.setChecked(self.tracker.drawProcessedFrame)
        self.wDrawProcessed.stateChanged.connect(self.setDrawProcessed)
        self.pGridSave.addRow(QLabel('Draw processed frame:'),self.wDrawProcessed)

        self.wNFrames = QLabel('')
        self.wNFrames.setMaximumHeight(25)
        self.wNFrames.setMaximumWidth(200)
        self.pGridSave.addRow(QLabel('Number of frames:'),self.wNFrames)

        self.saveParameters = QPushButton('Save tracker parameters')
        self.saveParameters.clicked.connect(self.saveTrackerParameters)
        self.pGridSave.addRow(self.saveParameters)

        
        self.tabDisplay.layout.addWidget(pGroup)

        #self.tabROI.layout = QFormLayout()
        #pGrid = self.tabROI.layout
        #self.tabROI.setLayout(self.tabROI.layout)
        
        
        #self.wROIscene = QGraphicsScene()
        #self.wROIview = QGraphicsView(self.wROIscene)
        #self.setROIImage(cv2.equalizeHist(image))
        #self.wROIview.setGeometry(0,0,75,75)
        #self.wROIview.scale(0.3,0.3)
        #self.wROIview.mouseReleaseEvent = self.selectPoints
        #pGrid.addRow(self.wROIview)
        self.setLayout(layout)
        self.show()

    def setROIImage(self,image):
        self.wROIscene.clear()
        if len(image.shape) == 2:
            frame  = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(frame, frame.shape[1], frame.shape[0], 
                               frame.strides[0], QImage.Format_RGB888)
        self.wROIscene.addPixmap(QPixmap.fromImage(qimage))
        self.wROIscene.update()
        
    def clearPoints(self,event):
        self.tracker.ROIpoints = []
        self.parameters['points'] = []
        self.tracker.parameters['pupilApprox'] = None
        self.update()
        
    def updateTrackerOutputBinaryImage(self,state):
        self.tracker.concatenateBinaryImage = state
        self.update()

    def setInvertThreshold(self,value):
        self.parameters['invertThreshold'] = value
        self.update()

    def setCRTrack(self,value):
        self.parameters['crTrack'] = value
        self.update()
            
    def setRoundIndex(self):
        value = self.wRoundIndex.toPlainText()
        try:
            self.parameters['roundIndex'] = float(value)
        except:
            print('Need to insert a float in the radius.')
        self.update()

    def setGamma(self,value):
        self.parameters['gamma'] = float(value)/10
        self.wGammaLabel.setText('Gamma [{0}]:'.format(self.parameters['gamma']))
        self.update()

    def setSequentialCrMode(self,value):
        self.parameters['sequentialCrMode'] = value

    def setSequentialPupilMode(self,value):
        self.parameters['sequentialPupilMode'] = value

    def setDrawProcessed(self,value):
        self.tracker.drawProcessedFrame = value
        self.update()

    def setBinThreshold(self,value):
        self.parameters['threshold'] = int(value)
        self.wBinThresholdLabel.setText('Binary contrast [{0}]:'.format(int(value)))
        self.update()

    def setContrastLim(self,value):
        self.parameters['contrast_clipLimit'] = int(value)
        self.wContrastLimLabel.setText('Contrast limit [{0}]:'.format(int(value)))
        self.tracker.set_clhe()
        self.update()

    def setContrastGridSize(self,value):
        self.parameters['contrast_gridSize'] = int(value)
        self.wContrastGridSizeLabel.setText('Contrast grid size [{0}]:'.format(int(value)))
        self.tracker.set_clhe()
        self.update()


    def setCloseKernelSize(self,value):
        self.parameters['close_kernelSize'] = int(value)
        self.wCloseKernelSizeLabel.setText('Morph close size [{0}]:'.format(int(value)))
        self.update()

    def setOpenKernelSize(self,value):
        self.parameters['open_kernelSize'] = int(value)
        self.wOpenKernelSizeLabel.setText('Morph open size [{0}]:'.format(int(value)))
        self.update()

    def resetSequentialPupil(self):
        self.tracker.parameters['pupilApprox'] = None

    def setEyeRadius(self):
        value = self.wEyeRadius.toPlainText()
        try:
            self.parameters['eye_radius_mm'] = float(value)
            print(self.tracker.parameters['eye_radius_mm'])
        except:
            print('Need to insert a float in the radius.')
    def setFOV(self):
        value = self.wFOV.toPlainText()
        try:
            self.parameters['fov_mm'] = float(value)
        except:
            pass
        '''
    def selectPoints(self,event):
        pt = self.wROIview.mapToScene(event.pos())
        if event.button() == 1:
            x = pt.x()
            y = pt.y()
            self.parameters['points'].append([int(round(x)),int(round(y))])
            #img = self.imgstack.get(int(self.wFrame.value()))
            #height,width = img.shape
            self.tracker.setROI(self.parameters['points'])
            self.putPoints()
        elif event.button() == 2:
            x = pt.x()
            y = pt.y()
            from .tracker import cropImageWithCoords
            img,(x1, y1, w, h) = cropImageWithCoords(self.tracker.ROIpoints, self.tracker.img)
            pts = [int(round(x))+x1,int(round(y))+y1]
            self.tracker.parameters['crApprox'] = pts
    '''
    def update(self):
        # This function can be overwritten to update the image
        pass

    def saveTrackerParameters(self,resultfile = None):
        if resultfile is None or resultfile == False:
            try:
                paramfile = QFileDialog().getSaveFileName()
            except:
                paramfile = ''
        else:
            paramfile = str(resultfile)
        if type(paramfile) is tuple:
            paramfile = paramfile[0]
        from .io import saveTrackerParameters
        res = saveTrackerParameters(paramfile,self.parameters)
        print('Saved parameters [{0}].'.format(paramfile))        

        
