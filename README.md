def refresh_ui(self):
        saved = [edit.text() for edit in self.shell_thickness_inputs]
        self.fig1.clear()
        self.fig2.clear()
        ax1, ax2 = self.fig1.add_subplot(111), self.fig2.add_subplot(111)

        for i in reversed(range(self.thickness_layout.count())):
            w = self.thickness_layout.itemAt(i).widget()
            if w: w.setParent(None)
        self.shell_thickness_inputs.clear()

        if self.is_calculated:
            # -------------------------------------------------------------
            # [1번째 창] 단면 1D 라인 그리기 (기존 동일)
            # -------------------------------------------------------------
            if self.centerlines:
                for cl in self.centerlines:
                    lo = cl['line']
                    x, y = lo.xy
                    ct = cl.get('type', '')
                    if ct == '1999': color = '#FF00FF'
                    elif ct == '157': color = '#00FFFF'
                    elif ct == '1102': color = '#FFA500'
                    elif ct == 'stiffener': color = '#008000'
                    elif ct == 'bridge': color = '#00CC00'
                    else: color = '#00FF00'
                    thk = cl.get('thickness', 10.0)

                    if thk > 0:
                        try:
                            poly = lo.buffer(thk / 2.0, cap_style=2)
                            if poly.geom_type == 'Polygon':
                                ax1.fill(*poly.exterior.xy, color=color, alpha=0.3, zorder=9, edgecolor='none')
                            elif poly.geom_type == 'MultiPolygon':
                                for p in poly.geoms:
                                    ax1.fill(*p.exterior.xy, color=color, alpha=0.3, zorder=9, edgecolor='none')
                        except:
                            pass
                    ax1.plot(x, y, color=color, linewidth=2.5, alpha=0.9, zorder=10, linestyle='-')

            # -------------------------------------------------------------
            # [2번째 창] Closed Cells (배경 처리)
            # -------------------------------------------------------------
            if hasattr(self, 'mesh_cells') and self.mesh_cells:
                cmap = matplotlib.colormaps.get_cmap('tab20')
                for idx, poly in enumerate(self.mesh_cells):
                    color = cmap(idx % 20)
                    ax2.fill(*poly.exterior.xy, color=color, alpha=0.15) # q를 잘 보이게 하기 위해 알파값 조정
                    ax2.plot(*poly.exterior.xy, color='gray', linewidth=0.5, alpha=0.5)

            # -------------------------------------------------------------
            # [추가] 2번째 창(ax2)에 전단류(q) 흐름 및 Max 전단응력(tau) 지점 시각화
            # -------------------------------------------------------------
            if hasattr(self, 'graph_edges') and self.graph_edges:
                q_cmap = plt.get_cmap("coolwarm")
                q_vals = [e.get('q_final_mean', 0.0) for e in self.graph_edges]
                max_q = max([abs(v) for v in q_vals], default=1.0)
                if max_q < 1e-12: max_q = 1.0

                for e in self.graph_edges:
                    line = e['line']
                    xs, ys = line.xy
                    q_val = e.get('q_final_mean', 0.0)
                    
                    # q의 크기에 비례한 색상 및 선 굵기 반영
                    color = q_cmap(0.5 + 0.5 * q_val / max_q)
                    lw = 1.5 + 4.0 * (abs(q_val) / max_q)
                    
                    ax2.plot(xs, ys, color=color, linewidth=lw, alpha=0.9, zorder=12)
                    
                    # 엣지 중앙에 q 값 텍스트 표시
                    mid = line.interpolate(0.5, normalized=True)
                    ax2.text(mid.x, mid.y, f"{q_val:.1f}", fontsize=7, color='black',
                             ha='center', va='center',
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.3),
                             zorder=13)

            # 최대 전단응력 지점 마커 표시
            if hasattr(self, 'max_tau_info') and self.max_tau_info.get('point'):
                mx = self.max_tau_info['x']
                my = self.max_tau_info['y']
                max_tau = self.max_tau_info['tau_abs']
                
                # 별 모양(★)으로 강렬하게 표시
                ax2.plot(mx, my, marker='*', color='red', markersize=20, 
                         markeredgecolor='black', markeredgewidth=1.5, 
                         zorder=20, label='Max Shear Stress')
                
                # 텍스트 라벨
                ax2.text(mx, my + 150, f"Max τ: {max_tau:.2f} MPa", color='red', 
                         fontsize=11, fontweight='bold', ha='center',
                         bbox=dict(facecolor='yellow', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.3'),
                         zorder=21)

            # Node Points 표시
            if hasattr(self, 'cell_points') and self.cell_points:
                ax2.plot([p[0] for p in self.cell_points], [p[1] for p in self.cell_points], 'ko', markersize=3,
                         markeredgecolor='white', markeredgewidth=0.5,
                         label=f'Nodes ({len(self.cell_points)})', zorder=11)

            h, l = ax2.get_legend_handles_labels()
            if h: ax2.legend(loc="upper right", fontsize=9)

        else:
            if self.raw_1999_lines:
                for ls in self.raw_1999_lines:
                    ax1.plot(*ls.xy, color='black', lw=1.5)
            for i, l in enumerate(self.left_1999_segments):
                ax1.text(l.centroid.x, l.centroid.y, f"S{i + 1}", fontsize=10, fontweight='bold',
                         ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 좌측 쉘 두께 입력칸 동적 갱신
        for i in range(len(self.left_1999_segments)):
            f = QFrame()
            r = QHBoxLayout(f)
            f.setStyleSheet("border-left: 4px solid #2E86C1; background: #FDFEFE; margin-bottom: 2px;")
            r.addWidget(QLabel(f"S{i + 1}:"))
            r.addStretch()
            edit = QLineEdit(saved[i] if i < len(saved) else "10")
            edit.setFixedWidth(self.field_width)
            edit.setStyleSheet(self.input_style)
            r.addWidget(edit)
            self.thickness_layout.addWidget(f)
            self.shell_thickness_inputs.append(edit)

        for ax in [ax1, ax2]:
            ax.set_aspect('equal')
            ax.grid(True, lw=0.3)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{-x:g}"))
        self.can1.draw()
        self.can2.draw()
