:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.9999991059303284).
nn('X0',1,2.884604244379231e-14).
nn('X0',2,8.888712841326196e-07).
nn('X0',3,1.8726788511818498e-13).
nn('X0',4,2.6521848834546976e-15).
nn('X0',5,2.36596315456028e-11).
nn('X0',6,6.213436987145826e-10).
nn('X0',7,7.622418668473685e-11).
nn('X0',8,1.6282349912799532e-09).
nn('X0',9,7.852644229311423e-11).
nn('X1',0,2.776368319246103e-06).
nn('X1',1,0.002024610759690404).
nn('X1',2,0.2499084770679474).
nn('X1',3,2.4780360035947524e-05).
nn('X1',4,6.376257601914403e-07).
nn('X1',5,3.6141081238838524e-08).
nn('X1',6,3.744195620214441e-09).
nn('X1',7,0.7068955898284912).
nn('X1',8,0.040655408054590225).
nn('X1',9,0.00048767117550596595).
nn('X2',0,1.1450647008359738e-07).
nn('X2',1,9.306944104281456e-11).
nn('X2',2,2.635945520523819e-06).
nn('X2',3,4.790542962318511e-13).
nn('X2',4,0.00033794782939366996).
nn('X2',5,2.950192356365733e-05).
nn('X2',6,0.9996297359466553).
nn('X2',7,3.622657285262676e-09).
nn('X2',8,4.479862869555262e-10).
nn('X2',9,1.5181769175143245e-09).
nn('X3',0,0.9999993443489075).
nn('X3',1,3.6297736192604513e-13).
nn('X3',2,3.625624600545052e-08).
nn('X3',3,2.804422547051516e-13).
nn('X3',4,5.9432265267445e-17).
nn('X3',5,3.781916099732108e-11).
nn('X3',6,6.488201035464991e-12).
nn('X3',7,5.97082191688969e-07).
nn('X3',8,1.8752756292261807e-11).
nn('X3',9,7.279944425064855e-10).

a :- Pos=[f(['X0','X1','X2','X3'],13)], metaabd(Pos).