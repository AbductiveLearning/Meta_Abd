:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.0016227546148002148).
nn('X0',1,0.009450416080653667).
nn('X0',2,0.1362510770559311).
nn('X0',3,0.0005415874184109271).
nn('X0',4,0.0026181975845247507).
nn('X0',5,0.003990820609033108).
nn('X0',6,0.5986471772193909).
nn('X0',7,7.230602932395414e-05).
nn('X0',8,0.24677133560180664).
nn('X0',9,3.434845348238014e-05).
nn('X1',0,7.979055993700968e-08).
nn('X1',1,0.9999998807907104).
nn('X1',2,5.383349321874675e-09).
nn('X1',3,2.4285670405615945e-17).
nn('X1',4,7.350471620259924e-11).
nn('X1',5,3.337921050672321e-09).
nn('X1',6,1.2183646092012168e-08).
nn('X1',7,7.851434502548216e-10).
nn('X1',8,1.8951973324021765e-10).
nn('X1',9,5.6164538297931443e-11).
nn('X2',0,1.0).
nn('X2',1,4.640541139065364e-17).
nn('X2',2,6.444468569899442e-11).
nn('X2',3,1.4835552787033028e-19).
nn('X2',4,4.7230972538663653e-20).
nn('X2',5,1.0814399133412134e-14).
nn('X2',6,1.2195996816621868e-11).
nn('X2',7,1.575831275906804e-14).
nn('X2',8,3.103934481402809e-13).
nn('X2',9,2.6582707622411605e-16).
nn('X3',0,4.820099319680082e-10).
nn('X3',1,1.355918328727057e-07).
nn('X3',2,7.644302968401462e-07).
nn('X3',3,0.005643158219754696).
nn('X3',4,0.004740152508020401).
nn('X3',5,0.00024033288354985416).
nn('X3',6,2.0190543681408712e-11).
nn('X3',7,0.5106796026229858).
nn('X3',8,3.9586058164786664e-07).
nn('X3',9,0.47869548201560974).
nn('X4',0,3.752326449557586e-08).
nn('X4',1,7.949367814034656e-17).
nn('X4',2,1.4855301710969826e-10).
nn('X4',3,1.8391374765730042e-19).
nn('X4',4,1.7317371714398178e-07).
nn('X4',5,2.413621132291155e-06).
nn('X4',6,0.9999973773956299).
nn('X4',7,8.052070955935934e-16).
nn('X4',8,3.605903282129924e-14).
nn('X4',9,7.275387059566419e-15).
nn('X5',0,9.117073415665367e-13).
nn('X5',1,3.9970257244901404e-17).
nn('X5',2,1.4798980183666766e-12).
nn('X5',3,9.480230489522512e-11).
nn('X5',4,3.950228347093798e-06).
nn('X5',5,6.664230500774693e-09).
nn('X5',6,2.4919736413275828e-14).
nn('X5',7,7.611719775013626e-05).
nn('X5',8,8.593304887116737e-12).
nn('X5',9,0.9999198913574219).
nn('X6',0,1.0).
nn('X6',1,3.906143824858761e-19).
nn('X6',2,6.53307297149297e-10).
nn('X6',3,3.686933417740293e-21).
nn('X6',4,1.1221189811639733e-23).
nn('X6',5,2.5668301145137106e-17).
nn('X6',6,1.4983323352109508e-14).
nn('X6',7,7.03043614264879e-15).
nn('X6',8,1.4996030116867743e-16).
nn('X6',9,1.874691567694742e-16).

a :- Pos=[f(['X0','X1','X2','X3'],16),f(['X4','X5','X6'],15)], metaabd(Pos).
