:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,4.007127699878765e-06).
nn('X0',1,0.00927647203207016).
nn('X0',2,0.0009216807666234672).
nn('X0',3,4.7128047242495086e-08).
nn('X0',4,0.9625925421714783).
nn('X0',5,0.002512597246095538).
nn('X0',6,0.00011940465628867969).
nn('X0',7,0.0007618266972713172).
nn('X0',8,6.385774759110063e-05).
nn('X0',9,0.02374754101037979).
nn('X1',0,9.738688744320712e-10).
nn('X1',1,4.484747080368834e-07).
nn('X1',2,0.9999995231628418).
nn('X1',3,2.3017753068149682e-12).
nn('X1',4,4.983656561881155e-21).
nn('X1',5,1.9622631277378119e-16).
nn('X1',6,1.1770075491096587e-15).
nn('X1',7,2.4511890295286776e-09).
nn('X1',8,1.2599312869052137e-13).
nn('X1',9,1.1101836673716113e-17).
nn('X2',0,2.375062494765212e-13).
nn('X2',1,1.962663143117993e-09).
nn('X2',2,1.6807196061563445e-06).
nn('X2',3,0.9999982714653015).
nn('X2',4,4.377521186832492e-22).
nn('X2',5,3.476873544627779e-08).
nn('X2',6,5.54718961102844e-22).
nn('X2',7,2.4034531418033744e-13).
nn('X2',8,2.0996340377684527e-17).
nn('X2',9,1.513510233889506e-17).
nn('X3',0,1.0).
nn('X3',1,7.89580600821547e-19).
nn('X3',2,1.5346926507397995e-10).
nn('X3',3,2.614173466000938e-17).
nn('X3',4,4.58681311629021e-23).
nn('X3',5,4.3928788829184634e-14).
nn('X3',6,9.445439222433016e-13).
nn('X3',7,1.642113501383255e-13).
nn('X3',8,1.7676872877785132e-14).
nn('X3',9,9.076493778168966e-16).
nn('X4',0,1.0).
nn('X4',1,1.9513552541643915e-18).
nn('X4',2,8.181975062748759e-13).
nn('X4',3,1.1781327961909342e-19).
nn('X4',4,9.245940875103758e-25).
nn('X4',5,2.1214091681237557e-12).
nn('X4',6,1.6320506144446023e-12).
nn('X4',7,5.273958827707419e-14).
nn('X4',8,4.036219527146373e-17).
nn('X4',9,2.1867724074606317e-19).
nn('X5',0,2.184823016948556e-10).
nn('X5',1,3.672251295938622e-06).
nn('X5',2,8.200385309464764e-06).
nn('X5',3,5.71316595596727e-05).
nn('X5',4,0.002396671799942851).
nn('X5',5,0.005271418951451778).
nn('X5',6,3.21980365924901e-07).
nn('X5',7,0.0013302513398230076).
nn('X5',8,0.9668485522270203).
nn('X5',9,0.024083750322461128).

a :- Pos=[f(['X0','X1','X2'],9),f(['X3','X4','X5'],8)], metaabd(Pos).
