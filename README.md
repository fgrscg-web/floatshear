import sys
import os
import io
import traceback
import datetime
import numpy as np
import pandas as pd
import ezdxf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import rcParams, cm
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QPushButton, QLabel, QTextEdit, QFileDialog, QLineEdit,
                               QHBoxLayout, QScrollArea, QFrame, QSplitter, QComboBox,
                               QInputDialog, QMessageBox, QProgressDialog, QDialog)
from PySide6.QtGui import QTextCursor, QIcon, QCloseEvent
from PySide6.QtCore import Qt

from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union, polygonize, split, nearest_points, snap
import shapely.affinity as affinity
from collections import defaultdict, deque

matplotlib.use('QtAgg')
rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False


class LoopViewerDialog(QDialog):
    """폐단면 검출 + CCW 순환 방향 화살표 표시"""
    def __init__(self, centerlines, loops, parent=None):
        super().__init__(parent)
        self.setWindowTitle("폐단면 검출 & 순환 방향 (CCW)")
        self.resize(1000, 700)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>회색: 1D 네트워크 / 색상 영역: 폐단면(Cell) / 화살표: CCW 순환 방향</b>"))
        self.fig = Figure(); canvas = FigureCanvas(self.fig)
        layout.addWidget(NavigationToolbar(canvas, self)); layout.addWidget(canvas, stretch=1)
        btn = QPushButton("닫기"); btn.setFixedHeight(40)
        btn.setStyleSheet("background-color:#8E44AD;color:white;font-weight:bold;font-size:14px;")
        btn.clicked.connect(self.close); layout.addWidget(btn)
        ax = self.fig.add_subplot(111); ax.set_aspect('equal'); ax.grid(True, linestyle=':', alpha=0.6)
        for cl in centerlines: ax.plot(*cl['line'].xy, color='gray', linewidth=1.5, alpha=0.6)
        cmap_c = matplotlib.colormaps.get_cmap('tab20')
        for idx, poly in enumerate(loops):
            color = cmap_c(idx % 20)
            ax.fill(*poly.exterior.xy, color=color, alpha=0.35)
            ax.plot(*poly.exterior.xy, color=color, linewidth=2)
            cx, cy = poly.centroid.x, poly.centroid.y
            ax.text(cx, cy, f"Cell {idx+1}", fontsize=10, fontweight='bold', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))
            # CCW 화살표
            coords = list(poly.exterior.coords)
            n = len(coords)
            step = max(1, n // 5)
            for i in range(0, n - 1, step):
                x1, y1 = coords[i]; x2, y2 = coords[min(i + 1, n - 2)]
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle='->', color=color, lw=2.5, mutation_scale=15))
        canvas.draw()

    def closeEvent(self, event: QCloseEvent):
        plt.close(self.fig); super().closeEvent(event)


class UltimateShipAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FloatCalc - Ship Floating Strength Analyzer")
        self.setWindowIcon(QIcon('icon.ico'))
        self.resize(2000, 1200)
        self.saved_frames_data = []; self.current_dxf_path = ""
        self.is_processing = False; self.debug_dialogs = []
        self.reset_analysis_data(); self.init_ui()

    def reset_analysis_data(self):
        self.raw_1999_lines = []; self.left_1999_segments = []
        self.lines_1102 = []; self.lines_1102_raw = []; self.lines_157 = []
        self.lines_6001 = []; self.lines_7001 = []; self.lines_9001 = []
        self.lines_minus1204 = []; self.hull_centroid = Point(0, 0)
        self.shell_thickness_inputs = []; self.calculated_polygons = []
        self.is_calculated = False; self.mesh_cells = []; self.centerlines = []; self.nodes = []
        self.shear_edges = []; self.max_tau = 0.0; self.max_tau_location = (0, 0); self.max_tau_thickness = 0.0
        self.calc_na_bl = 0.0; self.calc_depth = 0.0; self.base_report = ""
        self.act_fb = 0.0; self.act_fs = 0.0; self.allow_fb = 0.0; self.allow_fs = 0.0
        self.raw_swbm = 0.0; self.raw_shear = 0.0
        self.max_shell_q_idx = -1; self.max_shell_thk = 0.0; self.q_per_v = 0.0; self.max_Q = 0.0
        self.calc_max_q_val = 0.0; self.max_layer_name = ""
        for d in self.debug_dialogs: d.close()
        self.debug_dialogs.clear()

    def init_ui(self):
        main_scroll = QScrollArea(); main_scroll.setWidgetResizable(True)
        mc = QWidget(); ml = QHBoxLayout(mc)
        self.input_style = "background-color:white;color:black;border:1px solid #ABB2B9;padding:2px;"
        self.field_width = 100
        cp = QWidget(); cp.setFixedWidth(300); cpl = QVBoxLayout(cp); cpl.setAlignment(Qt.AlignTop)
        gb = QFrame(); gb.setStyleSheet("background:#F2F4F4;border-radius:5px;padding:5px;")
        gv = QVBoxLayout(gb); gv.addWidget(QLabel("<b>[General Settings]</b>"))
        for lbl, attr, dv in [("Scale:","txt_scale","100"),("H-Ext (mm):","txt_ext","10"),("V-Ext (mm):","txt_perp","10")]:
            h = QHBoxLayout(); h.addWidget(QLabel(lbl)); h.addStretch()
            le = QLineEdit(dv); le.setFixedWidth(self.field_width); le.setStyleSheet(self.input_style)
            setattr(self, attr, le); h.addWidget(le); gv.addLayout(h)
        cpl.addWidget(gb)
        self.btn_load = QPushButton("1. DXF Load 📂"); self.btn_load.setFixedHeight(40)
        self.btn_load.setStyleSheet("background-color:#2E86C1;color:white;font-weight:bold;margin-top:5px;margin-bottom:5px;")
        self.btn_load.clicked.connect(self.load_and_process_dxf); cpl.addWidget(self.btn_load)
        sb = QFrame(); sb.setStyleSheet("background:#FEF9E7;border-radius:5px;padding:5px;")
        sv = QVBoxLayout(sb); sv.addWidget(QLabel("<b>[Structural Category]</b>"))
        for lbl, attr, items in [("Continuity:","combo_section",["Continuous","Discontinuous"]),
                                  ("Hull Type:","combo_hull",["S/H","D/H"])]:
            h = QHBoxLayout(); h.addWidget(QLabel(lbl)); h.addStretch()
            cb = QComboBox(); cb.addItems(items); cb.setFixedWidth(self.field_width); cb.setStyleSheet(self.input_style)
            setattr(self, attr, cb); h.addWidget(cb); sv.addLayout(h)
        self.combo_section.currentTextChanged.connect(lambda t: self.combo_hull.setEnabled(t=="Discontinuous"))
        cpl.addWidget(sb)
        shb = QFrame(); shb.setStyleSheet("background:#EBF5FB;border-radius:5px;padding:5px;margin-top:10px;")
        shv = QVBoxLayout(shb); shv.addWidget(QLabel("<b>[Loading Setting]</b>"))
        for lbl, attr, dv in [("S.W.B.M (tm):","txt_swbm","0"),("Shear (t):","txt_shear_v","10"),("Grade (K):","txt_grade_k","1.00")]:
            h = QHBoxLayout(); h.addWidget(QLabel(lbl)); h.addStretch()
            le = QLineEdit(dv); le.setFixedWidth(self.field_width); le.setStyleSheet(self.input_style)
            setattr(self, attr, le); h.addWidget(le); shv.addLayout(h)
        cpl.addWidget(shb)
        cpl.addWidget(QLabel("<b>[S/SHELL Thickness (mm)]</b>"))
        self.thickness_scroll = QScrollArea(); self.thickness_scroll.setWidgetResizable(True); self.thickness_scroll.setMinimumHeight(200)
        self.scroll_content = QWidget(); self.thickness_layout = QVBoxLayout(self.scroll_content); self.thickness_layout.setAlignment(Qt.AlignTop)
        self.thickness_scroll.setWidget(self.scroll_content); cpl.addWidget(self.thickness_scroll)
        self.btn_calc = QPushButton("2. Cross Section Analysis 🧮"); self.btn_calc.setFixedHeight(50)
        self.btn_calc.setStyleSheet("background-color:#28B463;color:white;font-weight:bold;font-size:14px;margin-top:5px;")
        self.btn_calc.clicked.connect(self.calculate_total_inertia); cpl.addWidget(self.btn_calc)
        eb = QFrame(); eb.setStyleSheet("background:#FADBD8;border-radius:5px;padding:5px;margin-top:10px;")
        ev = QVBoxLayout(eb); ev.addWidget(QLabel("<b>[Post-Calc Evaluation]</b>"))
        hm = QHBoxLayout(); hm.addWidget(QLabel("Material:")); hm.addStretch()
        self.combo_material_type = QComboBox(); self.combo_material_type.addItems(["Mild","H.T","H.T with BKT"])
        self.combo_material_type.setFixedWidth(self.field_width); hm.addWidget(self.combo_material_type); ev.addLayout(hm)
        self.btn_eval = QPushButton("3. STRENGTH Analysis ⛓️"); self.btn_eval.setFixedHeight(40)
        self.btn_eval.setStyleSheet("background-color:#E67E22;color:white;font-weight:bold;")
        self.btn_eval.setEnabled(False); self.btn_eval.clicked.connect(self.evaluate_strength)
        ev.addWidget(self.btn_eval); cpl.addWidget(eb); ml.addWidget(cp)
        wa = QWidget(); wl = QVBoxLayout(wa); vs = QSplitter(Qt.Horizontal)
        for i, title in enumerate(["[Cross Section View]","[Shear Stress]"]):
            c = QWidget(); ly = QVBoxLayout(c); ly.addWidget(QLabel(f"<b>{title}</b>"))
            fig = Figure(); can = FigureCanvas(fig); ly.addWidget(NavigationToolbar(can, self)); ly.addWidget(can, stretch=1)
            setattr(self, f"fig{i+1}", fig); setattr(self, f"can{i+1}", can); vs.addWidget(c)
        wl.addWidget(vs, stretch=7)
        self.result_box = QTextEdit(); self.result_box.setReadOnly(True); self.result_box.setFixedHeight(270)
        self.result_box.setStyleSheet("font-family:'Consolas';font-size:13px;"); wl.addWidget(self.result_box); ml.addWidget(wa, stretch=7)
        hp = QWidget(); hp.setFixedWidth(300); hl = QVBoxLayout(hp); hl.setAlignment(Qt.AlignTop)
        hl.addWidget(QLabel("<b>[Saved Frames for Excel]</b>"))
        self.history_scroll = QScrollArea(); self.history_scroll.setWidgetResizable(True)
        self.history_content = QWidget(); self.history_list_layout = QVBoxLayout(self.history_content)
        self.history_list_layout.setAlignment(Qt.AlignTop); self.history_scroll.setWidget(self.history_content)
        hl.addWidget(self.history_scroll, stretch=1)
        self.btn_save_frame = QPushButton("4. Add Frame to List 💾"); self.btn_save_frame.setFixedHeight(40)
        self.btn_save_frame.setStyleSheet("background-color:#3498DB;color:white;font-weight:bold;")
        self.btn_save_frame.clicked.connect(self.save_current_frame); self.btn_save_frame.setEnabled(False); hl.addWidget(self.btn_save_frame)
        self.btn_export_excel = QPushButton("5. Export All to Excel 📊"); self.btn_export_excel.setFixedHeight(40)
        self.btn_export_excel.setStyleSheet("background-color:#1E8449;color:white;font-weight:bold;margin-top:5px;")
        self.btn_export_excel.clicked.connect(self.export_to_excel); hl.addWidget(self.btn_export_excel)
        ml.addWidget(hp); main_scroll.setWidget(mc); self.setCentralWidget(main_scroll)

    def _extract_pts(self, e, scale):
        try:
            if e.dxftype()=='LINE': return [(e.dxf.start.x*scale,e.dxf.start.y*scale),(e.dxf.end.x*scale,e.dxf.end.y*scale)]
            elif e.dxftype() in ('LWPOLYLINE','POLYLINE'): return [(p[0]*scale,p[1]*scale) for p in e.get_points()]
        except: return None
        return None

    def generate_outward_thickness(self, line, thickness):
        try:
            coords=list(line.coords)
            if len(coords)<2: return line.buffer(thickness/2.0)
            cx=sum(p[0] for p in coords)/len(coords); cy=sum(p[1] for p in coords)/len(coords)
            vx,vy=cx-self.hull_centroid.x,cy-self.hull_centroid.y
            lx,ly=coords[-1][0]-coords[0][0],coords[-1][1]-coords[0][1]
            length=(lx**2+ly**2)**0.5
            if length==0: return line.buffer(thickness/2.0)
            nx,ny=-ly/length,lx/length
            if (nx*vx+ny*vy)<0: nx,ny=-nx,-ny
            return Polygon(coords+[(p[0]+nx*thickness,p[1]+ny*thickness) for p in reversed(coords)])
        except: return line.buffer(thickness/2.0)

    def apply_original_algorithm(self, target_lines, shell_lines, ext, perp, max_a):
        if not target_lines: return []
        noded=unary_union(target_lines+shell_lines)
        all_l=list(noded.geoms) if noded.geom_type=='MultiLineString' else [noded] if noded.geom_type in ('LineString','LinearRing') else []
        ext_l,p_lines=[],[]
        all_p=[pt for l in all_l for pt in [l.coords[0],l.coords[-1]]]
        u_p,cnts=np.unique(np.round(all_p,3),axis=0,return_counts=True)
        ends=[tuple(p) for p,c in zip(u_p,cnts) if c==1]
        for pt in ends:
            for l in all_l:
                if np.allclose(l.coords[0],pt) or np.allclose(l.coords[-1],pt):
                    c=list(l.coords)
                    p_t,p_a=(np.array(c[0]),np.array(c[1])) if np.allclose(c[0],pt) else (np.array(c[-1]),np.array(c[-2]))
                    diff=p_t-p_a; norm=np.linalg.norm(diff)
                    if norm>1e-9:
                        vec=diff/norm; n=np.array([-vec[1],vec[0]])
                        p_lines.append(LineString([tuple(pt+n*perp),pt,tuple(pt-n*perp)])); break
        for l in all_l:
            c=list(l.coords); ps,pn=np.array(c[0]),np.array(c[1]); pl,pp=np.array(c[-1]),np.array(c[-2])
            vs=(ps-pn)/(np.linalg.norm(ps-pn)+1e-9); ve=(pl-pp)/(np.linalg.norm(pl-pp)+1e-9)
            c[0],c[-1]=tuple(ps+vs*ext),tuple(pl+ve*ext); ext_l.append(LineString(c))
        return [p for p in polygonize(unary_union(ext_l+p_lines+shell_lines)) if p.area<max_a]

    def heal_1102_collinear(self, lines, tg=150.0):
        if not lines: return []
        br=[]; gr={}
        for l in lines:
            c=list(l.coords); p1,p2=np.array(c[0]),np.array(c[-1]); v=p2-p1; L=np.linalg.norm(v)
            if L<1e-6: continue
            ak=round(np.degrees(np.arctan2(v[1],v[0]))%180,0)
            th=np.radians(ak); rho=round((-p1[0]*np.sin(th)+p1[1]*np.cos(th))/10)*10
            k=(ak,rho); gr.setdefault(k,[]).append((l,p1,p2))
        for (ak,_),g in gr.items():
            if len(g)<2: continue
            dv=np.array([np.cos(np.radians(ak)),np.sin(np.radians(ak))])
            segs=sorted([(np.dot(p1,dv),np.dot(p2,dv),p1,p2) for _,p1,p2 in g],key=lambda x:min(x[0],x[1]))
            for i in range(len(segs)-1):
                pe=segs[i][2] if segs[i][0]>segs[i][1] else segs[i][3]
                pn=segs[i+1][3] if segs[i+1][0]>segs[i+1][1] else segs[i+1][2]
                gap=np.linalg.norm(pn-pe)
                if 0.1<gap<=tg: br.append(LineString([tuple(pe),tuple(pn)]))
        return lines+br

    def heal_1999_collinear(self, infos, tg=500.0):
        if not infos: return []
        br=[]; gr={}
        for info in infos:
            c=list(info['line'].coords); p1,p2=np.array(c[0]),np.array(c[-1]); v=p2-p1; L=np.linalg.norm(v)
            if L<1e-6: continue
            ak=round(np.degrees(np.arctan2(v[1],v[0]))%180,0)
            th=np.radians(ak); rho=round((-p1[0]*np.sin(th)+p1[1]*np.cos(th))/10)*10
            k=(ak,rho); gr.setdefault(k,[]).append((info,p1,p2))
        for (ak,_),g in gr.items():
            if len(g)<2: continue
            dv=np.array([np.cos(np.radians(ak)),np.sin(np.radians(ak))])
            segs=sorted([(np.dot(p1,dv),np.dot(p2,dv),p1,p2,info) for info,p1,p2 in g],key=lambda x:min(x[0],x[1]))
            for i in range(len(segs)-1):
                pe=segs[i][2] if segs[i][0]>segs[i][1] else segs[i][3]
                pn=segs[i+1][3] if segs[i+1][0]>segs[i+1][1] else segs[i+1][2]
                gap=np.linalg.norm(pn-pe)
                if 0.1<gap<=tg:
                    br.append({'line':LineString([tuple(pe),tuple(pn)]),'thickness':(segs[i][4]['thickness']+segs[i+1][4]['thickness'])/2,'type':'1999','is_bridge':True})
        return infos+br

    def load_and_process_dxf(self):
        if self.is_processing: return
        fname,_=QFileDialog.getOpenFileName(self,'Select DXF File','','DXF files (*.dxf)')
        if not fname: return
        self.reset_analysis_data(); self.result_box.clear(); self.current_dxf_path=fname
        try:
            scale=float(self.txt_scale.text())
            try: doc=ezdxf.readfile(fname,encoding='cp949')
            except:
                try: doc=ezdxf.readfile(fname,encoding='utf-8')
                except: doc=ezdxf.readfile(fname)
            msp=doc.modelspace(); active={l.dxf.name for l in doc.layers if l.is_on() and not l.is_frozen()}
            t1999,t1204=[],[]; tl={"-1102":[],"157":[],"6001":[],"7001":[],"9001":[]}
            for e in msp:
                layer=e.dxf.layer.strip()
                if layer not in active: continue
                pts=self._extract_pts(e,scale)
                if not pts or len(pts)<2: continue
                ls=LineString(pts)
                if layer=="1999": t1999.append(ls)
                elif layer=="-1204": t1204.append(ls)
                elif layer in tl: tl[layer].append(ls)
            if t1999: u=unary_union(t1999); self.cx,self.cy_base=u.centroid.x,u.bounds[1]
            else: self.cx=self.cy_base=0.0
            shift=lambda ls:LineString([(p[0]-self.cx,p[1]-self.cy_base) for p in ls.coords])
            self.raw_1999_lines=[shift(ls) for ls in t1999]
            self.lines_minus1204=[shift(ls) for ls in t1204]
            m1999=unary_union(self.raw_1999_lines); self.hull_centroid=m1999.centroid
            cutters=[LineString([tuple(np.array(c.coords[0])-20),tuple(np.array(c.coords[-1])+20)]) for c in [shift(ls) for ls in t1204]]
            sr=split(m1999,unary_union(cutters)) if cutters else m1999
            pieces=list(sr.geoms) if hasattr(sr,'geoms') else [sr]
            self.left_1999_segments=sorted([g for g in pieces if g.centroid.x<=0.1 and g.length>0.1],key=lambda s:(-round(s.centroid.y,2),s.centroid.x))
            self.lines_1102=[shift(ls) for ls in tl["-1102"]]; self.lines_1102_raw=list(self.lines_1102)
            self.lines_157=[shift(ls) for ls in tl["157"]]
            self.lines_6001=[shift(ls) for ls in tl["6001"]]; self.lines_7001=[shift(ls) for ls in tl["7001"]]
            self.lines_9001=[shift(ls) for ls in tl["9001"]]
            self.refresh_ui(); self.result_box.append(f"✅ Loaded: {os.path.basename(fname)}")
        except Exception as e: self.result_box.setText(f"❌ Error:\n{traceback.format_exc()}")

    def calculate_total_inertia(self):
        if self.is_processing: return
        self.is_processing=True; self.btn_calc.setEnabled(False); self.btn_load.setEnabled(False)
        progress=QProgressDialog("Processing...","Cancel",0,100,self)
        progress.setWindowTitle("Processing"); progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False); progress.show(); QApplication.processEvents()

        def filter_short(lines,ml=100.0): return [l for l in lines if l.length>=ml]
        def remove_overlapping(lines,dt=10.0,at=5.0):
            lines=sorted(lines,key=lambda x:x.length,reverse=True); kept=[]
            for l in lines:
                c=list(l.coords); ps,pe=np.array(c[0]),np.array(c[-1]); v=pe-ps; ln=np.linalg.norm(v)
                if ln<1e-6: continue
                ang=np.degrees(np.arctan2(v[1],v[0]))%180; dup=False
                for k in kept:
                    ck=list(k.coords); pk1,pk2=np.array(ck[0]),np.array(ck[-1]); vk=pk2-pk1; lk=np.linalg.norm(vk)
                    if lk<1e-6: continue
                    ak=np.degrees(np.arctan2(vk[1],vk[0]))%180
                    if min(abs(ang-ak),180-abs(ang-ak))>at: continue
                    vu=vk/lk; mid=(ps+pe)/2.0
                    if np.linalg.norm(mid-(pk1+np.dot(mid-pk1,vu)*vu))>dt: continue
                    t1,t2=np.dot(ps-pk1,vu),np.dot(pe-pk1,vu)
                    if min(lk,max(t1,t2))-max(0,min(t1,t2))>ln*0.8: dup=True; break
                if not dup: kept.append(l)
            return kept
        def split_by_slope(line,at=5.0):
            coords=list(line.coords)
            if len(coords)<3: return [line]
            segs=[]; cur=[coords[0]]
            for i in range(1,len(coords)-1):
                cur.append(coords[i])
                v1=np.array(coords[i])-np.array(coords[i-1]); v2=np.array(coords[i+1])-np.array(coords[i])
                n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
                if n1<1e-6 or n2<1e-6: continue
                if np.degrees(np.arccos(np.clip(np.dot(v1,v2)/(n1*n2),-1,1)))>at: segs.append(LineString(cur)); cur=[coords[i]]
            cur.append(coords[-1])
            if len(cur)>=2: segs.append(LineString(cur))
            return segs
        def match_pairs(lines,md=100.0,at=20.0,ot=5.0):
            if not lines: return []
            ls_s=sorted(lines,key=lambda x:x.length); meta=[]
            for l in ls_s:
                c=list(l.coords); ps,pe=np.array(c[0]),np.array(c[-1]); v=pe-ps; ln=np.linalg.norm(v)
                if ln<1e-6: meta.append(None); continue
                meta.append({'ps':ps,'pe':pe,'ln':ln,'unit':v/ln,'ang':np.degrees(np.arctan2(v[1],v[0]))%180,'mid':(ps+pe)/2})
            used={i:[] for i in range(len(ls_s))}; pairs=[]
            for i in range(len(ls_s)):
                if meta[i] is None: continue
                mi=meta[i]; bj,bd,bo=-1,float('inf'),None
                for j in range(i+1,len(ls_s)):
                    if meta[j] is None: continue
                    mj=meta[j]; ad=min(abs(mi['ang']-mj['ang']),180-abs(mi['ang']-mj['ang']))
                    if ad>at: continue
                    d=np.linalg.norm(mi['mid']-(mj['ps']+np.dot(mi['mid']-mj['ps'],mj['unit'])*mj['unit']))
                    if d>md: continue
                    t1=np.dot(mi['ps']-mj['ps'],mj['unit']); t2=np.dot(mi['pe']-mj['ps'],mj['unit'])
                    os_,oe_=max(0,min(t1,t2)),min(mj['ln'],max(t1,t2))
                    if (oe_-os_)<mi['ln']*0.1: continue
                    blocked=any(min(oe_,ue)-max(os_,us)>ot for us,ue in used[j])
                    if blocked: continue
                    if d<bd: bd,bj,bo=d,j,(os_,oe_)
                if bj>=0 and bo: used[bj].append(bo); pairs.append((i,bj,bo,bd))
            return [(ls_s[i],ls_s[j],ov,dist) for i,j,ov,dist in pairs]
        def create_centerlines(pairs):
            result=[]
            for sl,ll,(os_,oe_),dist in pairs:
                cs=list(sl.coords); cl_c=list(ll.coords); ps1,ps2=np.array(cs[0]),np.array(cs[-1])
                pl1=np.array(cl_c[0]); vl=np.array(cl_c[-1])-pl1; ln=np.linalg.norm(vl)
                if ln<1e-6: continue
                vl_u=vl/ln; mids=[]
                for f in np.linspace(0,1,5):
                    pt_s=ps1+(ps2-ps1)*f; t=np.dot(pt_s-pl1,vl_u); mids.append(tuple((pt_s+pl1+t*vl_u)/2))
                result.append({'line':LineString(mids),'thickness':round(dist*2)/2.0})
            return result
        def raycast_extend(cls,md=100.0):
            ext_pts=[]; result=[]
            for i,cl in enumerate(cls):
                coords=list(cl['line'].coords)
                if len(coords)<2: result.append(cl); continue
                for ei in [0,-1]:
                    p=np.array(coords[ei]); nb=1 if ei==0 else -2; v=p-np.array(coords[nb]); vn=np.linalg.norm(v)
                    if vn<1e-6: continue
                    d=v/vn; conn=False
                    for j,o in enumerate(cls):
                        if i==j: continue
                        for op in [list(o['line'].coords)[0],list(o['line'].coords)[-1]]:
                            if np.linalg.norm(p-np.array(op))<5.0: conn=True; break
                        if conn: break
                    if conn: continue
                    ray=LineString([tuple(p),tuple(p+d*md)]); bp_,bd_=None,md
                    for j,o in enumerate(cls):
                        if i==j: continue
                        inter=ray.intersection(o['line'])
                        if inter.is_empty: continue
                        pts=[inter] if inter.geom_type=='Point' else list(inter.geoms) if inter.geom_type=='MultiPoint' else []
                        for pt in pts:
                            dd=np.linalg.norm(np.array([pt.x,pt.y])-p)
                            if 1e-3<dd<bd_: bd_=dd; bp_=(pt.x,pt.y)
                    if bp_:
                        bp_o=(bp_[0]+d[0]*0.1,bp_[1]+d[1]*0.1)
                        if ei==0: coords[0]=bp_o
                        else: coords[-1]=bp_o
                        ext_pts.append(np.array(bp_o))
                result.append({'line':LineString(coords),'thickness':cl['thickness'],'type':cl.get('type','internal')})
            return result,ext_pts
        def bridge_open_nodes(cls,ext_pts,mg=100.0):
            ep=defaultdict(list)
            for idx,cl in enumerate(cls):
                coords=list(cl['line'].coords)
                for side,pt in [('start',coords[0]),('end',coords[-1])]:
                    k=(round(pt[0],1),round(pt[1],1))
                    ep[k].append({'idx':idx,'pt':np.array(pt),'thk':cl['thickness'],'type':cl.get('type','')})
            opens=[]
            for k,conns in ep.items():
                if len(conns)!=1: continue
                info=conns[0]
                if any(np.linalg.norm(info['pt']-ep_)<1.0 for ep_ in ext_pts): continue
                opens.append(info)
            matched=set(); bridges=[]
            for i,e1 in enumerate(opens):
                if i in matched: continue
                bj,bd=-1,float('inf')
                for j,e2 in enumerate(opens):
                    if j<=i or j in matched or e1['idx']==e2['idx']: continue
                    d=np.linalg.norm(e1['pt']-e2['pt'])
                    if 0.1<d<=mg and d<bd: bd=d; bj=j
                if bj>=0:
                    e2=opens[bj]; mid=tuple((e1['pt']+e2['pt'])/2)
                    bridges.append({'line':LineString([tuple(e1['pt']),mid]),'thickness':e1['thk'],'type':'bridge'})
                    bridges.append({'line':LineString([mid,tuple(e2['pt'])]),'thickness':e2['thk'],'type':'bridge'})
                    matched.add(i); matched.add(bj)
            return cls+bridges,len(bridges)//2
        def filter_redundant_nodes(cls):
            ep=defaultdict(list)
            for idx,cl in enumerate(cls):
                coords=list(cl['line'].coords)
                ep[(round(coords[0][0],1),round(coords[0][1],1))].append((idx,'start'))
                ep[(round(coords[-1][0],1),round(coords[-1][1],1))].append((idx,'end'))
            merged=set(); mp=[]
            for k,conns in ep.items():
                if len(conns)!=2: continue
                ia,sa=conns[0]; ib,sb=conns[1]
                if ia==ib or ia in merged or ib in merged: continue
                if abs(cls[ia]['thickness']-cls[ib]['thickness'])>0.5: continue
                if cls[ia].get('type')!=cls[ib].get('type'): continue
                mp.append((ia,sa,ib,sb)); merged.add(ia); merged.add(ib)
            nl=[]
            for ia,sa,ib,sb in mp:
                ca=list(cls[ia]['line'].coords); cb=list(cls[ib]['line'].coords)
                if sa=='start': ca=ca[::-1]
                if sb=='end': cb=cb[::-1]
                nl.append({'line':LineString(ca+cb[1:]),'thickness':cls[ia]['thickness'],'type':cls[ia].get('type','internal')})
            for idx,cl in enumerate(cls):
                if idx not in merged: nl.append(cl)
            return nl
        def split_at_intersections(cls):
            geoms=[cl['line'] for cl in cls]; ipts=[]
            for i in range(len(geoms)):
                for j in range(i+1,len(geoms)):
                    try:
                        inter=geoms[i].intersection(geoms[j])
                        if inter.is_empty: continue
                        if inter.geom_type=='Point': ipts.append(inter)
                        elif inter.geom_type=='MultiPoint': ipts.extend(inter.geoms)
                        elif inter.geom_type=='GeometryCollection': ipts.extend([g for g in inter.geoms if g.geom_type=='Point'])
                    except: pass
            for g in geoms: ipts.append(Point(g.coords[0])); ipts.append(Point(g.coords[-1]))
            if not ipts: return cls
            upts=[]
            for pt in ipts:
                if not upts or min(pt.distance(u) for u in upts)>1e-3: upts.append(pt)
            splitter=unary_union(upts); ncls=[]
            for cl in cls:
                try:
                    res=split(snap(cl['line'],splitter,0.05),splitter)
                    for g in (list(res.geoms) if hasattr(res,'geoms') else [res]):
                        ncls.append({'line':g,'thickness':cl['thickness'],'type':cl.get('type','internal')})
                except: ncls.append(cl)
            return ncls

        try:
            # ============ [code 1] 이너시아 ============
            temp_left,temp_right,input_thks=[],[],[]
            for i,ls in enumerate(self.left_1999_segments):
                if progress.wasCanceled(): raise UserWarning("Canceled")
                try: t=float(self.shell_thickness_inputs[i].text())
                except: t=10.0
                input_thks.append(t); lp=self.generate_outward_thickness(ls,t)
                temp_left.append(lp); temp_right.append(affinity.scale(lp,xfact=-1.0,origin=(0,0)))
            shells=temp_left+temp_right
            if not shells: raise UserWarning("No shells")
            ym=unary_union(shells).bounds[1]
            temp_left=[affinity.translate(p,yoff=-ym) for p in temp_left]
            temp_right=[affinity.translate(p,yoff=-ym) for p in temp_right]
            self.lines_1102=[affinity.translate(l,yoff=-ym) for l in self.lines_1102]
            self.lines_157=[affinity.translate(l,yoff=-ym) for l in self.lines_157]
            self.lines_6001=[affinity.translate(l,yoff=-ym) for l in self.lines_6001]
            self.lines_7001=[affinity.translate(l,yoff=-ym) for l in self.lines_7001]
            self.lines_9001=[affinity.translate(l,yoff=-ym) for l in self.lines_9001]
            self.raw_1999_lines=[affinity.translate(l,yoff=-ym) for l in self.raw_1999_lines]
            c1102=[affinity.translate(l,yoff=-ym) for l in self.lines_1102_raw]
            c157=list(self.lines_157)
            l1999s=[affinity.translate(l,yoff=-ym) for l in self.left_1999_segments]
            ext,perp=float(self.txt_ext.text()),float(self.txt_perp.text())
            internal=self.lines_1102+self.lines_157+self.lines_6001+self.lines_7001+self.lines_9001
            ipoly=self.apply_original_algorithm(internal,self.raw_1999_lines,ext,perp,0.5e6) if internal else []
            self.calculated_polygons=temp_left+temp_right+ipoly
            if not self.calculated_polygons: raise UserWarning("No polygons")
            self.calc_total_area=sum(p.area for p in self.calculated_polygons)
            self.calc_na_bl=sum(p.centroid.y*p.area for p in self.calculated_polygons)/(self.calc_total_area or 1e-9)
            self.calc_ixx=0.0
            def rixx(ring,na):
                pts=list(ring.coords); v=0.0
                for i in range(len(pts)-1):
                    x1,y1=pts[i][0],pts[i][1]-na; x2,y2=pts[i+1][0],pts[i+1][1]-na
                    v+=(y1**2+y1*y2+y2**2)*(x1*y2-x2*y1)
                return abs(v/12.0)
            for poly in self.calculated_polygons:
                self.calc_ixx+=rixx(poly.exterior,self.calc_na_bl)
                for h in poly.interiors: self.calc_ixx-=rixx(h,self.calc_na_bl)
            self.raw_swbm=float(self.txt_swbm.text()); self.raw_shear=float(self.txt_shear_v.text())

            # ============ [code 2] 1D 추출 ============
            progress.setLabelText("1D Extraction..."); QApplication.processEvents()
            l1999f=[]
            for i,ls in enumerate(l1999s):
                try: ta=float(self.shell_thickness_inputs[i].text())
                except: ta=10.0
                l1999f.append({'line':ls,'thickness':ta,'type':'1999'})
                l1999f.append({'line':affinity.scale(ls,xfact=-1.0,origin=(0,0)),'thickness':ta,'type':'1999'})
            f1102=remove_overlapping(filter_short(c1102,100),dt=1.0)
            f157=remove_overlapping(filter_short(c157,100),dt=1.0)
            h1102=self.heal_1102_collinear(f1102,150)
            l1999f=self.heal_1999_collinear(l1999f,500)
            s157=[]
            for l in f157: s157.extend(split_by_slope(l,at=5.0))
            s157=[s for s in s157 if s.length>=30.0]
            p1102=match_pairs(h1102,100,20,5); p157=match_pairs(s157,100,20,5)
            cl1102=create_centerlines(p1102)
            for cl in cl1102: cl['type']='1102'
            cl157=create_centerlines(p157)
            for cl in cl157: cl['type']='157'
            cl1999=[l for l in l1999f if l['line'].length>100 or l.get('is_bridge',False)]
            all_cl=cl1999+cl1102+cl157
            all_cl,ext_pts=raycast_extend(all_cl,100)
            all_cl,bc=bridge_open_nodes(all_cl,ext_pts,100)
            prev=-1
            for _ in range(10):
                nl=filter_redundant_nodes(all_cl)
                if len(nl)==prev: break
                prev=len(nl); all_cl=nl
            all_cl=split_at_intersections(all_cl)
            nodes_set=set()
            for cl in all_cl:
                coords=list(cl['line'].coords); nodes_set.add(tuple(coords[0])); nodes_set.add(tuple(coords[-1]))
            extracted_nodes=list(nodes_set)

            # ============ Step 10: 폐루프 검출 ============
            progress.setLabelText("Detecting cells..."); QApplication.processEvents()
            planar=unary_union([cl['line'] for cl in all_cl])
            try:
                from shapely import set_precision
                planar=set_precision(planar,grid_size=0.01)
            except: pass
            detected_loops=[p for p in polygonize(planar) if p.area>=100.0]
            # 면적 큰 순 정렬
            detected_loops=sorted(detected_loops,key=lambda p:p.area,reverse=True)
            self.mesh_cells=detected_loops
            dlg=LoopViewerDialog(all_cl,detected_loops,self); dlg.show(); self.debug_dialogs.append(dlg)

            # ============ Step 11~17: 전단응력 계산 ============
            progress.setLabelText("Computing shear stress..."); QApplication.processEvents()
            V_n=abs(self.raw_shear)*9806.65  # N

            # 11. 토폴로지 그래프
            node_to_id={}
            for i,pt in enumerate(extracted_nodes):
                node_to_id[(round(pt[0],1),round(pt[1],1))]=i
            g_nodes={i:{'coord':extracted_nodes[i],'edges':[]} for i in range(len(extracted_nodes))}
            g_edges=[]
            for eid,cl in enumerate(all_cl):
                coords=list(cl['line'].coords)
                sk=(round(coords[0][0],1),round(coords[0][1],1))
                ek=(round(coords[-1][0],1),round(coords[-1][1],1))
                sn=node_to_id.get(sk,-1); en=node_to_id.get(ek,-1)
                edge={'id':eid,'start_node':sn,'end_node':en,'line':cl['line'],'length':cl['line'].length,
                      'thickness':cl['thickness'],'type':cl.get('type',''),'left_cell':None,'right_cell':None}
                g_edges.append(edge)
                if sn>=0: g_nodes[sn]['edges'].append(eid)
                if en>=0: g_nodes[en]['edges'].append(eid)

            # 12. 엣지-셀 매핑
            for edge in g_edges:
                coords=list(edge['line'].coords); p1,p2=np.array(coords[0]),np.array(coords[-1])
                mid=(p1+p2)/2.0; d=p2-p1; dn=np.linalg.norm(d)
                if dn<1e-6: continue
                normal=np.array([-d[1],d[0]])/dn
                pt_l=Point(mid+normal*1.0); pt_r=Point(mid-normal*1.0)
                for cid,cpoly in enumerate(detected_loops):
                    if cpoly.contains(pt_l): edge['left_cell']=cid
                    if cpoly.contains(pt_r): edge['right_cell']=cid

            # 13. σ 테이블
            sigma={cid:{} for cid in range(len(detected_loops))}
            for edge in g_edges:
                if edge['left_cell'] is not None: sigma[edge['left_cell']][edge['id']]=+1
                if edge['right_cell'] is not None: sigma[edge['right_cell']][edge['id']]=-1

            # 14. 샘플링 + S_local
            for edge in g_edges:
                L=edge['length']; ns=max(2,int(L/100)+1)
                edge['sample_s']=np.linspace(0,L,ns)
                edge['sample_pts']=[]; edge['sample_y']=[]
                for s in edge['sample_s']:
                    pt=edge['line'].interpolate(s); edge['sample_pts'].append((pt.x,pt.y)); edge['sample_y'].append(pt.y)
                t=edge['thickness']; yb=np.array(edge['sample_y'])-self.calc_na_bl
                ds=np.diff(edge['sample_s']); Sl=np.zeros(ns)
                for k in range(1,ns): Sl[k]=Sl[k-1]+t*(yb[k-1]+yb[k])/2.0*ds[k-1]
                edge['S_local']=Sl; edge['S_total']=Sl[-1]

            # 15. BFS → S 누적 → q₀
            best_start=0; best_x=float('inf')
            for nid,nd in g_nodes.items():
                if abs(nd['coord'][0])<best_x: best_x=abs(nd['coord'][0]); best_start=nid
            S_node={nid:None for nid in g_nodes}; S_node[best_start]=0.0
            visited_e=set(); queue=deque([best_start])
            while queue:
                nid=queue.popleft()
                for eid in g_nodes[nid]['edges']:
                    if eid in visited_e: continue
                    edge=g_edges[eid]; visited_e.add(eid)
                    if edge['start_node']==nid:
                        s_at_start=S_node[nid]; edge['S_accumulated']=s_at_start+edge['S_local']
                        exit_n=edge['end_node']; s_exit=s_at_start+edge['S_total']
                    else:
                        s_at_start=S_node[nid]-edge['S_total']; edge['S_accumulated']=s_at_start+edge['S_local']
                        exit_n=edge['start_node']; s_exit=s_at_start
                    if exit_n>=0 and S_node.get(exit_n) is None: S_node[exit_n]=s_exit; queue.append(exit_n)
            # 미방문 엣지 처리
            for edge in g_edges:
                if 'S_accumulated' not in edge:
                    sn,en=edge['start_node'],edge['end_node']
                    if sn>=0 and S_node.get(sn) is not None:
                        edge['S_accumulated']=S_node[sn]+edge['S_local']
                    elif en>=0 and S_node.get(en) is not None:
                        edge['S_accumulated']=(S_node[en]-edge['S_total'])+edge['S_local']
                    else:
                        edge['S_accumulated']=np.zeros(len(edge['sample_s']))
            # q₀
            for edge in g_edges:
                edge['q0']=-V_n*edge['S_accumulated']/self.calc_ixx if self.calc_ixx!=0 else np.zeros(len(edge['sample_s']))

            # 16. 순환 전단류 [A]{qc}={b}
            nc=len(detected_loops)
            if nc>0:
                A=np.zeros((nc,nc)); b=np.zeros(nc)
                for ci in range(nc):
                    for eid,sig_i in sigma[ci].items():
                        edge=g_edges[eid]; t=edge['thickness']
                        if t<=0: continue
                        A[ci][ci]+=edge['length']/t
                        oc=edge['right_cell'] if edge['left_cell']==ci else edge['left_cell'] if edge['right_cell']==ci else None
                        if oc is not None: A[ci][oc]-=edge['length']/t
                        q0v=edge['q0']; dsv=np.diff(edge['sample_s']); qi=0.0
                        for k in range(len(dsv)): qi+=(q0v[k]+q0v[k+1])/2.0*dsv[k]/t
                        b[ci]-=sig_i*qi
                try: qc=np.linalg.solve(A,b)
                except: qc=np.zeros(nc)
            else: qc=np.array([])

            # 17. 최종 q, τ
            self.max_tau=0.0; self.max_tau_location=(0,0); self.max_tau_thickness=0.0
            for edge in g_edges:
                qf=np.copy(edge['q0'])
                if edge['left_cell'] is not None and edge['left_cell']<nc:
                    qf+=sigma[edge['left_cell']][edge['id']]*qc[edge['left_cell']]
                if edge['right_cell'] is not None and edge['right_cell']<nc:
                    qf+=sigma[edge['right_cell']][edge['id']]*qc[edge['right_cell']]
                edge['q_final']=qf
                t=edge['thickness']; edge['tau']=qf/t if t>0 else np.zeros_like(qf)
                if t>0:
                    idx_=np.argmax(np.abs(edge['tau']))
                    if np.abs(edge['tau'][idx_])>self.max_tau:
                        self.max_tau=np.abs(edge['tau'][idx_])
                        self.max_tau_location=edge['sample_pts'][idx_]
                        self.max_tau_thickness=t
            self.shear_edges=g_edges
            self.act_fs=self.max_tau
            self.centerlines=all_cl; self.nodes=extracted_nodes

            # ============ 보고서 ============
            all_y=[pt[1] for p in self.calculated_polygons for pt in p.exterior.coords]
            y_max,y_min=max(all_y),min(all_y)
            self.calc_depth=(y_max-y_min)*1e-3
            dt_=y_max-self.calc_na_bl; db_=self.calc_na_bl-y_min
            zt=self.calc_ixx/dt_ if dt_!=0 else 1e-9
            self.calc_z_top=zt*1e-9; self.calc_z_btm=(self.calc_ixx/db_*1e-9) if db_!=0 else 0
            self.act_fb=(abs(self.raw_swbm)*9.80665e6)/zt
            res =f"--- Applied Loads ---\nS.W.B.M: {self.raw_swbm:,.2f} tm\nShear: {self.raw_shear:,.2f} t\n\n"
            res+=f"--- Geometric Properties ---\nArea: {self.calc_total_area/100:>10,.2f} cm²\n"
            res+=f"I_xx: {self.calc_ixx*1e-12:,.6e} m⁴\nDepth: {self.calc_depth:.3f} m\n"
            res+=f"N.A from B.L: {self.calc_na_bl*1e-3:.3f} m\nZ_btm: {self.calc_z_btm:,.4f} m³\nZ_top: {self.calc_z_top:,.4f} m³\n\n"
            res+=f"--- 1D Results ---\nCenterlines: {len(self.centerlines)} | Nodes: {len(self.nodes)} | Cells: {len(detected_loops)}\n\n"
            res+=f"--- Shear Stress ---\nMax τ: {self.max_tau:.2f} N/mm² at ({self.max_tau_location[0]:.0f}, {self.max_tau_location[1]:.0f})\n"
            res+=f"Thickness at max τ: {self.max_tau_thickness:.1f} mm\n"
            self.base_report=res; self.result_box.setText(res)
            self.is_calculated=True; self.btn_eval.setEnabled(True)
            progress.setLabelText("Rendering..."); progress.setMaximum(0); QApplication.processEvents()
            self.refresh_ui()
        except UserWarning as uw: self.result_box.setText(f"⚠️ {uw}")
        except Exception as e:
            self.result_box.setText(f"❌ Error:\n{e}\n\n{traceback.format_exc()}")
            QMessageBox.critical(self,"Error",str(e))
        finally:
            progress.close(); self.is_processing=False; self.btn_calc.setEnabled(True); self.btn_load.setEnabled(True)

    def evaluate_strength(self):
        if not self.is_calculated: return
        k=float(self.txt_grade_k.text()); self.allow_fs=105/k
        if self.combo_section.currentText()=="Continuous": self.allow_fb=143/k
        else:
            ht=self.combo_hull.currentText(); mt=self.combo_material_type.currentText()
            table={"S/H":{"Mild":60,"H.T":75,"H.T with BKT":112},"D/H":{"Mild":112,"H.T":150,"H.T with BKT":157}}
            self.allow_fb=table.get(ht,{}).get(mt,60)
        r=self.base_report+"\n--- Strength Check ---\n"
        r+=f"Material: {self.combo_material_type.currentText()}\n\n"
        r+=f"Bending: {self.act_fb:.2f} / {self.allow_fb:.2f} N/mm² ({self.act_fb/self.allow_fb*100:.1f}%) [{'PASS' if self.act_fb<=self.allow_fb else 'FAIL'}]\n"
        r+=f"Shear  : {self.act_fs:.2f} / {self.allow_fs:.2f} N/mm² ({self.act_fs/self.allow_fs*100:.1f}%) [{'PASS' if self.act_fs<=self.allow_fs else 'FAIL'}]\n"
        self.result_box.setText(r); self.btn_save_frame.setEnabled(True)

    def save_current_frame(self):
        fn,ok=QInputDialog.getText(self,"Save Frame","Name:",QLineEdit.EchoMode.Normal,"FR.")
        if not ok or not fn.strip(): return
        i1,i2=io.BytesIO(),io.BytesIO()
        self.fig1.savefig(i1,format='png',bbox_inches='tight',dpi=150)
        self.fig2.savefig(i2,format='png',bbox_inches='tight',dpi=150)
        self.saved_frames_data.append({
            "Frame":fn.strip(),"SWBM":self.raw_swbm,"Depth":round(self.calc_depth,2),
            "NA":round(self.calc_na_bl*1e-3,2),"Ixx":round(self.calc_ixx*1e-12,2),
            "Grade_K":float(self.txt_grade_k.text()),"Z_btm":round(self.calc_z_btm,2),
            "Z_top":round(self.calc_z_top,2),"Act_FB":round(self.act_fb,1),"Allow_FB":round(self.allow_fb,1),
            "Shear":self.raw_shear,"Pos_Shear":"N/A","Thk":self.max_tau_thickness,"Unit_q":0.0,
            "Act_FS":round(self.act_fs,1),"Allow_FS":round(self.allow_fs,1),
            "ImgSec":i1.getvalue(),"ImgShr":i2.getvalue()})
        self.update_history_list_ui()

    def update_history_list_ui(self):
        for i in reversed(range(self.history_list_layout.count())):
            w=self.history_list_layout.itemAt(i).widget()
            if w: w.deleteLater()
        for i,d in enumerate(self.saved_frames_data):
            f=QFrame(); f.setStyleSheet("background:white;border:1px solid #BDC3C7;border-radius:4px;margin-bottom:2px;")
            l=QHBoxLayout(f); l.setContentsMargins(5,5,5,5)
            lb=QLabel(f"📝 {d['Frame']}"); lb.setStyleSheet("font-weight:bold;border:none;"); l.addWidget(lb); l.addStretch()
            for t,h,e in [("✏️",lambda c=False,x=i:self.rename_frame(x),True),("⬆️",lambda c=False,x=i:self.move_frame_up(x),i>0),
                          ("⬇️",lambda c=False,x=i:self.move_frame_down(x),i<len(self.saved_frames_data)-1),
                          ("❌",lambda c=False,x=i:self.delete_saved_frame(x),True)]:
                b=QPushButton(t); b.setFixedSize(24,24); b.setStyleSheet("border:none;background:transparent;")
                b.setEnabled(e); b.clicked.connect(h); l.addWidget(b)
            self.history_list_layout.addWidget(f)
    def rename_frame(self,i):
        n,ok=QInputDialog.getText(self,"Rename","Name:",QLineEdit.Normal,self.saved_frames_data[i]['Frame'])
        if ok and n.strip(): self.saved_frames_data[i]['Frame']=n.strip(); self.update_history_list_ui()
    def move_frame_up(self,i):
        if i>0: self.saved_frames_data[i-1],self.saved_frames_data[i]=self.saved_frames_data[i],self.saved_frames_data[i-1]; self.update_history_list_ui()
    def move_frame_down(self,i):
        if i<len(self.saved_frames_data)-1: self.saved_frames_data[i+1],self.saved_frames_data[i]=self.saved_frames_data[i],self.saved_frames_data[i+1]; self.update_history_list_ui()
    def delete_saved_frame(self,i):
        if 0<=i<len(self.saved_frames_data): del self.saved_frames_data[i]; self.update_history_list_ui()

    def apply_outer_border(self,ws,r1,r2,c1,c2):
        from openpyxl.styles import Border,Side
        tk=Side(border_style="medium",color="000000")
        for r in range(r1,r2+1):
            for c in range(c1,c2+1):
                cl=ws.cell(row=r,column=c); b=cl.border
                cl.border=Border(top=tk if r==r1 else b.top,bottom=tk if r==r2 else b.bottom,left=tk if c==c1 else b.left,right=tk if c==c2 else b.right)

    def export_to_excel(self):
        if not self.saved_frames_data: return
        path,_=QFileDialog.getSaveFileName(self,"Export","Section_Analysis_Report.xlsx","Excel (*.xlsx)")
        if not path: return
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Alignment,Font,PatternFill,Border,Side
            from openpyxl.utils import get_column_letter
            from openpyxl.drawing.image import Image as XlImage
            from openpyxl.worksheet.pagebreak import Break
            wb=Workbook(); ws=wb.active; ws.title="Strength Result"
            ft=Font(name='돋움',size=14,bold=True,underline='double'); f11b=Font(name='돋움',size=11,bold=True)
            f11n=Font(name='돋움',size=11); f10n=Font(name='돋움',size=10); f18b=Font(name='돋움',size=18,bold=True)
            ac=Alignment(horizontal='center',vertical='center',wrap_text=True)
            fy=PatternFill(start_color='FFF2CC',end_color='FFF2CC',fill_type='solid')
            fg=PatternFill(start_color='E2EFDA',end_color='E2EFDA',fill_type='solid')
            ts=Side(border_style="thin",color="000000"); ba=Border(top=ts,left=ts,right=ts,bottom=ts)
            ws.merge_cells('A1:I4'); ws.cell(1,1,"H0000 Conclusion of Scantling Check for Partial Floating Condition").font=ft
            ws.cell(1,1).alignment=ac; ws.merge_cells('J1:J4'); ws.cell(1,10,"검 토").alignment=ac
            ws.cell(1,11,"PART장").alignment=ac; ws.cell(1,12,"보임과장").alignment=ac
            ws.merge_cells('K2:K3'); ws.merge_cells('L2:L3')
            ws.row_dimensions[2].height=ws.row_dimensions[3].height=30
            td=datetime.datetime.now().strftime("%Y.%m.%d."); ws.cell(4,11,td).alignment=ac; ws.cell(4,12,td).alignment=ac
            for r in range(1,5):
                for c in range(1,13): ws.cell(r,c).border=ba
            ws.cell(5,1,"*Allowable stress").font=f11b
            ws.merge_cells('A6:A8'); ws.cell(6,1,"Bending Stress")
            ws.merge_cells('B6:C6'); ws.cell(6,2,"Continuous Section"); ws.merge_cells('D6:L6'); ws.cell(6,4,"143/k")
            ws.merge_cells('B7:C8'); ws.cell(7,2,"Discontinuous Section")
            ws.merge_cells('D7:F7'); ws.cell(7,4,"60 S/H Mild"); ws.merge_cells('G7:I7'); ws.cell(7,7,"75 S/H H.T")
            ws.merge_cells('J7:L7'); ws.cell(7,10,"112 S/H H.T+BKT")
            ws.merge_cells('D8:F8'); ws.cell(8,4,"112 D/H Mild"); ws.merge_cells('G8:I8'); ws.cell(8,7,"150 D/H H.T")
            ws.merge_cells('J8:L8'); ws.cell(8,10,"157 D/H H.T+BKT")
            ws.merge_cells('A9:C9'); ws.cell(9,1,"Shear Stress"); ws.merge_cells('D9:L9'); ws.cell(9,4,"105/k")
            for r in range(6,10):
                for c in range(1,13): ws.cell(r,c).border=ba; ws.cell(r,c).alignment=ac
            ri=11
            for d in self.saved_frames_data:
                hd=["Position","S.W.B.M\n(t·m)","Depth(m)","N.A\nfrom B.L(m)","I_xx(m⁴)","Grade(k)",
                    "Z_btm\n(m³)","Z_top\n(m³)","σ_bend\n(N/mm²)","σ_allow\n(N/mm²)","Pct\n(%)","Result"]
                for i,h in enumerate(hd,1): c=ws.cell(ri,i,h); c.alignment=ac; c.font=f10n; c.border=ba
                r2=[d['Frame'],d['SWBM'],d['Depth'],d['NA'],d['Ixx'],d['Grade_K'],d['Z_btm'],d['Z_top'],
                    d['Act_FB'],d['Allow_FB'],d['Act_FB']/d['Allow_FB'] if d['Allow_FB']>0 else 0,
                    "OK" if d['Act_FB']<=d['Allow_FB'] else "NG"]
                for i,v in enumerate(r2,1):
                    c=ws.cell(ri+1,i,v); c.alignment=ac; c.border=ba; c.font=f11n
                    if i in [2,3,4,5,6]: c.fill=fy
                    if i in [9,10,11]: c.fill=fg
                    if i==2: c.number_format='#,##0'
                    elif i==6: c.number_format='0.00'
                    elif i==11: c.number_format='0%'
                    elif i in [3,4,5,7,8]: c.number_format='#,##0.00'
                ws.cell(ri+2,2,"SHEAR(t)").alignment=ac; ws.merge_cells(start_row=ri+2,start_column=3,end_row=ri+2,end_column=4)
                ws.cell(ri+2,3,"Position").alignment=ac; ws.cell(ri+2,5,"Thk(mm)").alignment=ac
                ws.cell(ri+2,6,"Grade(k)").alignment=ac; ws.merge_cells(start_row=ri+2,start_column=7,end_row=ri+2,end_column=8)
                ws.cell(ri+2,7,"q(N/mm)").alignment=ac; ws.cell(ri+2,9,"τ(N/mm²)").alignment=ac
                ws.cell(ri+2,10,"τ_allow").alignment=ac; ws.cell(ri+2,11,"Pct(%)").alignment=ac; ws.cell(ri+2,12,"Result").alignment=ac
                r4=[d['Shear'],d['Pos_Shear'],d['Thk'],d['Grade_K'],d['Unit_q'],d['Act_FS'],d['Allow_FS'],
                    d['Act_FS']/d['Allow_FS'] if d['Allow_FS']>0 else 0,"OK" if d['Act_FS']<=d['Allow_FS'] else "NG"]
                ws.cell(ri+3,2,r4[0]).fill=fy; ws.merge_cells(start_row=ri+3,start_column=3,end_row=ri+3,end_column=4)
                ws.cell(ri+3,3,r4[1]); ws.cell(ri+3,5,r4[2]).fill=fy; ws.cell(ri+3,6,r4[3]).fill=fy
                ws.merge_cells(start_row=ri+3,start_column=7,end_row=ri+3,end_column=8); ws.cell(ri+3,7,r4[4]).fill=fy
                ws.cell(ri+3,9,r4[5]).fill=fg; ws.cell(ri+3,10,r4[6]).fill=fg; ws.cell(ri+3,11,r4[7]).fill=fg
                ws.cell(ri+3,11).number_format='0%'; ws.cell(ri+3,12,r4[8])
                for r in range(ri+2,ri+4):
                    for c in range(2,13): ws.cell(r,c).border=ba; ws.cell(r,c).alignment=ac; ws.cell(r,c).font=f11n
                ws.merge_cells(start_row=ri+1,start_column=1,end_row=ri+3,end_column=1)
                for r in range(ri+1,ri+4): ws.cell(r,1).border=ba
                ws.cell(ri+1,1).font=f11b; ri+=4
            self.apply_outer_border(ws,1,ri-1,1,12)
            for i,w in enumerate([14,13,10,16,13,10,12,12,18,16,12,12],1): ws.column_dimensions[get_column_letter(i)].width=w
            ws.page_setup.orientation=ws.ORIENTATION_PORTRAIT; ws.page_setup.fitToPage=True
            ws.page_setup.fitToWidth=1; ws.page_setup.fitToHeight=0; ws.sheet_view.view='pageBreakPreview'
            ws.print_area=f"A1:L{ri-1}"; ws.page_setup.paperSize=ws.PAPERSIZE_A4
            ws.page_margins.left=0.3; ws.page_margins.right=0.3; ws.print_options.horizontalCentered=True
            wv=wb.create_sheet(title="Visualizations"); wv.sheet_view.view='pageBreakPreview'
            wv.page_setup.orientation=wv.ORIENTATION_LANDSCAPE; wv.page_setup.fitToPage=True
            wv.page_setup.fitToWidth=0; wv.page_setup.fitToHeight=1
            cp=2
            for d in self.saved_frames_data:
                wv.cell(2,cp,f"Results: {d['Frame']}").font=f18b
                im1=XlImage(io.BytesIO(d['ImgSec'])); im1.width,im1.height=int(im1.width*0.5),int(im1.height*0.5)
                wv.add_image(im1,wv.cell(4,cp).coordinate)
                im2=XlImage(io.BytesIO(d['ImgShr'])); im2.width,im2.height=int(im2.width*0.5),int(im2.height*0.5)
                wv.add_image(im2,wv.cell(22,cp).coordinate)
                wv.col_breaks.append(Break(id=((cp-2)//8+1)*8)); cp+=8
            wv.print_area=f"A1:{get_column_letter(cp-1)}40"
            for i in range(1,cp): wv.column_dimensions[get_column_letter(i)].width=9
            wb.save(path); QMessageBox.information(self,"Success","Export Done!")
        except PermissionError: QMessageBox.critical(self,"Error","File is open in Excel.")
        except Exception as e: QMessageBox.critical(self,"Error",str(e))

    def refresh_ui(self):
        saved=[e.text() for e in self.shell_thickness_inputs]
        self.fig1.clear(); self.fig2.clear()
        ax1,ax2=self.fig1.add_subplot(111),self.fig2.add_subplot(111)
        for i in reversed(range(self.thickness_layout.count())):
            w=self.thickness_layout.itemAt(i).widget()
            if w: w.setParent(None)
        self.shell_thickness_inputs.clear()
        if self.is_calculated:
            for poly in self.calculated_polygons:
                ax1.fill(*poly.exterior.xy,color='blue',alpha=0.4); ax1.plot(*poly.exterior.xy,color='darkblue',lw=1)
            # 전단응력 컬러맵
            if self.shear_edges:
                tau_all=np.concatenate([np.abs(e['tau']) for e in self.shear_edges if e['thickness']>0 and len(e['tau'])>0])
                vmax=np.max(tau_all) if len(tau_all)>0 and np.max(tau_all)>0 else 1.0
                norm=mcolors.Normalize(vmin=0,vmax=vmax); cmap_=cm.jet
                for edge in self.shear_edges:
                    if edge['thickness']<=0: continue
                    pts=edge['sample_pts']; tv=np.abs(edge['tau'])
                    for k in range(len(pts)-1):
                        c=cmap_(norm((tv[k]+tv[k+1])/2.0))
                        ax2.plot([pts[k][0],pts[k+1][0]],[pts[k][1],pts[k+1][1]],color=c,linewidth=5,solid_capstyle='round',zorder=10)
                sm=cm.ScalarMappable(cmap=cmap_,norm=norm); sm.set_array([])
                self.fig2.colorbar(sm,ax=ax2,label='|τ| (N/mm²)',shrink=0.8)
                # 최대 전단응력 지점
                if self.max_tau>0:
                    ax2.plot(self.max_tau_location[0],self.max_tau_location[1],'r*',markersize=20,
                             markeredgecolor='black',markeredgewidth=1,zorder=20)
                    ax2.annotate(f'τ_max={self.max_tau:.1f} N/mm²\nt={self.max_tau_thickness:.1f} mm',
                                xy=self.max_tau_location,xytext=(20,20),textcoords='offset points',
                                fontsize=9,fontweight='bold',arrowprops=dict(arrowstyle='->',color='red'),
                                bbox=dict(facecolor='yellow',alpha=0.9,edgecolor='red'),zorder=21)
            if self.nodes:
                ax2.plot([p[0] for p in self.nodes],[p[1] for p in self.nodes],'ko',markersize=3,alpha=0.5,zorder=15)
        else:
            if self.raw_1999_lines:
                for ls in self.raw_1999_lines: ax1.plot(*ls.xy,color='black',lw=1.5)
            for i,l in enumerate(self.left_1999_segments):
                ax1.text(l.centroid.x,l.centroid.y,f"S{i+1}",fontsize=10,fontweight='bold',ha='center',va='center',
                         bbox=dict(facecolor='white',alpha=0.7,edgecolor='none'))
        for i in range(len(self.left_1999_segments)):
            f=QFrame(); r=QHBoxLayout(f)
            f.setStyleSheet("border-left:4px solid #2E86C1;background:#FDFEFE;margin-bottom:2px;")
            r.addWidget(QLabel(f"S{i+1}:")); r.addStretch()
            e=QLineEdit(saved[i] if i<len(saved) else "10"); e.setFixedWidth(self.field_width); e.setStyleSheet(self.input_style)
            r.addWidget(e); self.thickness_layout.addWidget(f); self.shell_thickness_inputs.append(e)
        for ax in [ax1,ax2]:
            ax.set_aspect('equal'); ax.grid(True,lw=0.3)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{-x:g}"))
        self.can1.draw(); self.can2.draw()

if __name__=="__main__":
    app=QApplication(sys.argv); win=UltimateShipAnalyzer(); win.show(); sys.exit(app.exec())
