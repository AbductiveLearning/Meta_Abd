:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,3.8541014997506284e-12).
nn('X0',1,7.449607153375837e-08).
nn('X0',2,7.731476330263831e-07).
nn('X0',3,0.999991774559021).
nn('X0',4,8.819877801434437e-18).
nn('X0',5,7.457092124241171e-06).
nn('X0',6,9.986913840461001e-18).
nn('X0',7,1.4819322852188677e-11).
nn('X0',8,2.4006237214166956e-15).
nn('X0',9,6.577680938715368e-15).
nn('X1',0,0.0020620347931981087).
nn('X1',1,0.07967565953731537).
nn('X1',2,0.040796805173158646).
nn('X1',3,0.7907764911651611).
nn('X1',4,0.005445682909339666).
nn('X1',5,0.03266547620296478).
nn('X1',6,0.002413522219285369).
nn('X1',7,0.017558038234710693).
nn('X1',8,0.025131510570645332).
nn('X1',9,0.0034747463651001453).
nn('X2',0,7.39492204314884e-15).
nn('X2',1,5.892549339603663e-16).
nn('X2',2,3.933905610119248e-15).
nn('X2',3,1.6755126799883335e-11).
nn('X2',4,3.3180792526035145e-16).
nn('X2',5,1.0).
nn('X2',6,5.539889289596703e-16).
nn('X2',7,9.035477227525845e-11).
nn('X2',8,1.2118021802752076e-14).
nn('X2',9,9.019256869136072e-11).
nn('X3',0,1.1221608247069526e-06).
nn('X3',1,0.9999986886978149).
nn('X3',2,1.2731139520383294e-07).
nn('X3',3,4.751714544264736e-16).
nn('X3',4,1.768868940210666e-09).
nn('X3',5,2.8563500453770985e-09).
nn('X3',6,3.5430151257287434e-08).
nn('X3',7,3.530102876680985e-09).
nn('X3',8,1.1358248563908546e-08).
nn('X3',9,1.4820158433792585e-09).

a :- Pos=[f(['X0','X1','X2','X3'],12)], metaabd(Pos).