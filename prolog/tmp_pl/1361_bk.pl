:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,3.87065739460013e-07).
nn('X0',1,0.9999995231628418).
nn('X0',2,2.4555701472195324e-08).
nn('X0',3,9.227717754887635e-19).
nn('X0',4,2.428452838126871e-12).
nn('X0',5,2.796318732123382e-10).
nn('X0',6,1.994632903290494e-08).
nn('X0',7,5.502876279650515e-11).
nn('X0',8,4.720430318329305e-13).
nn('X0',9,1.6942720284466445e-13).
nn('X1',0,5.866810592891625e-09).
nn('X1',1,7.045831296181859e-08).
nn('X1',2,0.9999998807907104).
nn('X1',3,8.736853471594253e-13).
nn('X1',4,1.0616670514396446e-24).
nn('X1',5,1.8482612942391006e-17).
nn('X1',6,1.295686711777998e-15).
nn('X1',7,3.80546372191759e-10).
nn('X1',8,1.3493443558679208e-13).
nn('X1',9,1.1420733825009394e-18).
nn('X2',0,3.379743418463477e-08).
nn('X2',1,1.4058686940293663e-13).
nn('X2',2,4.823780264118227e-13).
nn('X2',3,1.1394380231011247e-11).
nn('X2',4,5.017896701994297e-14).
nn('X2',5,1.3749637162163708e-07).
nn('X2',6,2.285444445946852e-16).
nn('X2',7,0.9999911785125732).
nn('X2',8,3.3443174851611676e-14).
nn('X2',9,8.73722729011206e-06).
nn('X3',0,6.87832709900249e-07).
nn('X3',1,0.9999979734420776).
nn('X3',2,1.7107149119510723e-07).
nn('X3',3,1.0038899643212765e-13).
nn('X3',4,9.372057574452697e-10).
nn('X3',5,3.73845310264187e-08).
nn('X3',6,2.459703551949133e-09).
nn('X3',7,1.2239139550729305e-06).
nn('X3',8,6.491883142167865e-10).
nn('X3',9,1.3036554058487582e-09).
nn('X4',0,9.474058515479555e-07).
nn('X4',1,2.0516646714874298e-10).
nn('X4',2,4.9297440796181036e-08).
nn('X4',3,2.7788766843173107e-08).
nn('X4',4,4.1499919056775525e-09).
nn('X4',5,4.718441743989388e-07).
nn('X4',6,1.628350557690561e-12).
nn('X4',7,0.997324526309967).
nn('X4',8,7.294320258921516e-09).
nn('X4',9,0.0026740299072116613).

a :- Pos=[f(['X0','X1','X2'],10),f(['X3','X4'],8)], metaabd(Pos).
