:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,4.0072700474742817e-13).
nn('X0',1,3.787736183730184e-14).
nn('X0',2,4.282110148239038e-13).
nn('X0',3,6.505269656287282e-10).
nn('X0',4,2.1326237650831636e-11).
nn('X0',5,1.0).
nn('X0',6,1.110125121168981e-12).
nn('X0',7,4.005201859058616e-08).
nn('X0',8,1.259595509492395e-12).
nn('X0',9,4.4542641575162634e-08).
nn('X1',0,9.5632177732341e-09).
nn('X1',1,1.0).
nn('X1',2,4.247445797855853e-11).
nn('X1',3,2.1578440919820624e-19).
nn('X1',4,3.4092993185519394e-12).
nn('X1',5,3.088590216915321e-10).
nn('X1',6,8.147282327952698e-12).
nn('X1',7,4.736942504912633e-10).
nn('X1',8,1.0333673207494481e-13).
nn('X1',9,1.7133816067327334e-12).
nn('X2',0,2.3367007088381797e-05).
nn('X2',1,1.3109694536536654e-12).
nn('X2',2,1.3646193897098868e-10).
nn('X2',3,4.646207463904206e-14).
nn('X2',4,3.516405500114672e-10).
nn('X2',5,0.014924985356628895).
nn('X2',6,0.9850503206253052).
nn('X2',7,1.3785143816377743e-12).
nn('X2',8,1.261152419829159e-06).
nn('X2',9,2.5338030285038116e-12).
nn('X3',0,1.4715159757372476e-08).
nn('X3',1,1.0).
nn('X3',2,3.278230117920167e-11).
nn('X3',3,3.5785030348135996e-22).
nn('X3',4,5.817590875180356e-13).
nn('X3',5,1.2803498712632422e-11).
nn('X3',6,4.130860584150575e-11).
nn('X3',7,3.0615020067681353e-12).
nn('X3',8,8.782590709647143e-15).
nn('X3',9,2.5243702798678075e-14).
nn('X4',0,1.1076036754431584e-09).
nn('X4',1,1.5262957120398823e-08).
nn('X4',2,2.2295560029306216e-06).
nn('X4',3,1.1208007855145752e-09).
nn('X4',4,0.9929296374320984).
nn('X4',5,0.00022487255046144128).
nn('X4',6,1.4933982583897887e-06).
nn('X4',7,5.550889181904495e-05).
nn('X4',8,1.5176210288458947e-09).
nn('X4',9,0.006786306854337454).
nn('X5',0,7.26374120357462e-15).
nn('X5',1,3.747320768981597e-15).
nn('X5',2,3.398641435013554e-14).
nn('X5',3,3.5319884794660084e-09).
nn('X5',4,1.3622544656755053e-06).
nn('X5',5,7.412430136355397e-08).
nn('X5',6,5.095704550312058e-17).
nn('X5',7,0.00014470165478996933).
nn('X5',8,1.2633503965187032e-10).
nn('X5',9,0.9998538494110107).

a :- Pos=[f(['X0','X1','X2','X3'],13),f(['X4','X5'],13)], metaabd(Pos).
