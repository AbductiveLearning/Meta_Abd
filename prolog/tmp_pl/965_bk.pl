:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.7752197489784294e-09).
nn('X0',1,2.3947956950820526e-09).
nn('X0',2,4.581245452439653e-11).
nn('X0',3,1.7065140411887114e-07).
nn('X0',4,9.281336550009955e-13).
nn('X0',5,0.9999997615814209).
nn('X0',6,1.6340341302267802e-09).
nn('X0',7,2.2012429212736606e-08).
nn('X0',8,2.0361421437797844e-09).
nn('X0',9,1.0822220203321464e-10).
nn('X1',0,2.324064436720879e-14).
nn('X1',1,8.83608094614537e-17).
nn('X1',2,2.2919388063721282e-17).
nn('X1',3,4.739722679449886e-19).
nn('X1',4,6.497582875373593e-21).
nn('X1',5,6.885574196317107e-14).
nn('X1',6,4.1149464797798764e-25).
nn('X1',7,1.0).
nn('X1',8,2.3882386514509075e-22).
nn('X1',9,2.295565126875232e-10).
nn('X2',0,8.515969179834215e-11).
nn('X2',1,6.568956586697289e-22).
nn('X2',2,1.790237676526675e-14).
nn('X2',3,3.206398858461634e-25).
nn('X2',4,2.521443470682172e-11).
nn('X2',5,1.5155687549395225e-07).
nn('X2',6,0.9999998807907104).
nn('X2',7,2.9243472099745608e-21).
nn('X2',8,7.341902254043474e-17).
nn('X2',9,2.3216691510027502e-20).
nn('X3',0,7.903093290906327e-09).
nn('X3',1,1.4218591786629986e-05).
nn('X3',2,3.714204649440944e-05).
nn('X3',3,1.572906041524824e-11).
nn('X3',4,0.9990580677986145).
nn('X3',5,0.0007818056037649512).
nn('X3',6,6.2739286477153655e-06).
nn('X3',7,2.8661679607466795e-07).
nn('X3',8,2.0109212073293747e-08).
nn('X3',9,0.00010214487701887265).
nn('X4',0,1.9681023304130019e-13).
nn('X4',1,1.5448098800597415e-14).
nn('X4',2,3.4112116343437693e-12).
nn('X4',3,1.5930764618588e-07).
nn('X4',4,3.166051101288758e-07).
nn('X4',5,1.7307231203744777e-09).
nn('X4',6,7.808567194808689e-15).
nn('X4',7,0.0008592061931267381).
nn('X4',8,2.6604730862800352e-08).
nn('X4',9,0.9991403222084045).

a :- Pos=[f(['X0','X1','X2'],18),f(['X3','X4'],13)], metaabd(Pos).
