:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.4046157466307552e-11).
nn('X0',1,3.5447694239966276e-21).
nn('X0',2,8.126008885418236e-15).
nn('X0',3,2.6933115476656636e-24).
nn('X0',4,9.816933724260402e-12).
nn('X0',5,6.868861532893789e-07).
nn('X0',6,0.9999992847442627).
nn('X0',7,1.2809358366947902e-19).
nn('X0',8,4.555057647586174e-17).
nn('X0',9,2.6786954898996955e-20).
nn('X1',0,4.892742921924764e-08).
nn('X1',1,0.0005662093171849847).
nn('X1',2,5.386009797803126e-05).
nn('X1',3,0.9369112253189087).
nn('X1',4,2.5390539803993306e-07).
nn('X1',5,0.06244984641671181).
nn('X1',6,1.4216162691482737e-09).
nn('X1',7,6.774857411073754e-06).
nn('X1',8,2.9764187559067068e-08).
nn('X1',9,1.1643509424175136e-05).
nn('X2',0,6.743753999671753e-08).
nn('X2',1,0.9999998807907104).
nn('X2',2,2.831442813544527e-08).
nn('X2',3,1.3337738129490836e-17).
nn('X2',4,1.2451019382186956e-11).
nn('X2',5,8.469273615752115e-10).
nn('X2',6,5.321585394568729e-09).
nn('X2',7,1.2631847789990047e-09).
nn('X2',8,4.702123690414339e-10).
nn('X2',9,1.4924113334036093e-10).
nn('X3',0,0.00013635099458042532).
nn('X3',1,0.9998082518577576).
nn('X3',2,1.9375956981093623e-05).
nn('X3',3,7.953540048788454e-09).
nn('X3',4,1.0243137467114138e-06).
nn('X3',5,2.2225735847314354e-06).
nn('X3',6,2.917743415764562e-07).
nn('X3',7,3.089284655288793e-05).
nn('X3',8,2.3444103192105104e-07).
nn('X3',9,1.3247401966509642e-06).
nn('X4',0,9.05476860157961e-10).
nn('X4',1,1.5619683281162255e-13).
nn('X4',2,4.5588804953311524e-10).
nn('X4',3,2.7889123899171864e-08).
nn('X4',4,9.409914127900265e-06).
nn('X4',5,1.4725050334618572e-07).
nn('X4',6,1.50336704890508e-11).
nn('X4',7,0.0007536939810961485).
nn('X4',8,1.0151862994689509e-07).
nn('X4',9,0.9992365837097168).
nn('X5',0,7.549574299048345e-09).
nn('X5',1,2.6888069882039477e-15).
nn('X5',2,1.1702772084731805e-10).
nn('X5',3,9.159269675892699e-17).
nn('X5',4,5.507446720898201e-10).
nn('X5',5,1.0500963071535807e-05).
nn('X5',6,0.9999895691871643).
nn('X5',7,1.8648363764111296e-14).
nn('X5',8,7.981702972170712e-12).
nn('X5',9,5.916266791522375e-16).
nn('X6',0,1.1574171310257952e-07).
nn('X6',1,0.0002889269089791924).
nn('X6',2,0.9997105598449707).
nn('X6',3,1.9962538566620225e-12).
nn('X6',4,7.032601555908666e-17).
nn('X6',5,1.5865043992619766e-14).
nn('X6',6,1.0523853923721793e-12).
nn('X6',7,3.7751061654489604e-07).
nn('X6',8,2.5848786652793443e-11).
nn('X6',9,1.0020783980163248e-13).

a :- Pos=[f(['X0','X1','X2'],10),f(['X3','X4','X5','X6'],18)], metaabd(Pos).
