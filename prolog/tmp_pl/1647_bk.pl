:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,5.381449409355699e-13).
nn('X0',1,5.093812524559205e-12).
nn('X0',2,1.8903308085516102e-10).
nn('X0',3,2.1646587300416797e-11).
nn('X0',4,2.885981384426195e-13).
nn('X0',5,5.81717323200337e-08).
nn('X0',6,1.4252787838842096e-09).
nn('X0',7,1.9200049905521155e-07).
nn('X0',8,0.9999996423721313).
nn('X0',9,6.79777372170065e-08).
nn('X1',0,1.0).
nn('X1',1,4.028952486971153e-20).
nn('X1',2,1.0454961202333024e-10).
nn('X1',3,5.417263083978836e-22).
nn('X1',4,7.076476894729832e-29).
nn('X1',5,4.98326952924242e-16).
nn('X1',6,5.894983394656717e-14).
nn('X1',7,1.7178413152446408e-16).
nn('X1',8,8.9544710058809e-17).
nn('X1',9,1.2233490330228186e-21).
nn('X2',0,7.499196819082954e-10).
nn('X2',1,1.7103932714590542e-09).
nn('X2',2,5.155254001465437e-08).
nn('X2',3,3.9391153450196725e-08).
nn('X2',4,1.6934984664596087e-11).
nn('X2',5,2.6459804833289802e-11).
nn('X2',6,7.02345955577873e-17).
nn('X2',7,0.9996574521064758).
nn('X2',8,1.1800023458352626e-10).
nn('X2',9,0.00034259044332429767).
nn('X3',0,0.0008583090384490788).
nn('X3',1,0.9958778619766235).
nn('X3',2,0.00015225668903440237).
nn('X3',3,1.7985258864428033e-06).
nn('X3',4,1.69194408954354e-05).
nn('X3',5,0.00270792655646801).
nn('X3',6,0.0003522657498251647).
nn('X3',7,3.090068275923841e-05).
nn('X3',8,6.073420308894129e-07).
nn('X3',9,1.1333828524584533e-06).
nn('X4',0,3.578814761900917e-12).
nn('X4',1,3.798527714593547e-08).
nn('X4',2,9.445000728192099e-07).
nn('X4',3,0.9999849200248718).
nn('X4',4,2.385532770670176e-16).
nn('X4',5,1.4122520042292308e-05).
nn('X4',6,1.0025504297578438e-17).
nn('X4',7,1.0838028252635468e-09).
nn('X4',8,3.015990235333277e-14).
nn('X4',9,3.1747128255259216e-13).
nn('X5',0,1.0).
nn('X5',1,2.702370103271857e-15).
nn('X5',2,4.697411348786318e-09).
nn('X5',3,2.65394315179963e-17).
nn('X5',4,2.1125550158355598e-15).
nn('X5',5,2.5513613444161365e-10).
nn('X5',6,4.7667367830683816e-09).
nn('X5',7,2.4571347560786716e-12).
nn('X5',8,8.962588158613338e-13).
nn('X5',9,5.65245007917714e-13).

a :- Pos=[f(['X0','X1'],8),f(['X2','X3'],8),f(['X4','X5'],3)], metaabd(Pos).
