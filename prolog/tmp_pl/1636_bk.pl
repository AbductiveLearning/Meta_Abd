:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.1030214068508126e-11).
nn('X0',1,1.1359229290519579e-07).
nn('X0',2,3.9860387914814055e-06).
nn('X0',3,6.729443953190639e-07).
nn('X0',4,2.7583242356854498e-08).
nn('X0',5,1.1144952622998971e-05).
nn('X0',6,1.1116073892480927e-06).
nn('X0',7,3.275133713032119e-05).
nn('X0',8,0.9999448657035828).
nn('X0',9,5.358162525226362e-06).
nn('X1',0,1.1655388500587775e-14).
nn('X1',1,3.295750934718299e-16).
nn('X1',2,2.4325841803407317e-15).
nn('X1',3,2.931517528725891e-10).
nn('X1',4,8.973478623740634e-18).
nn('X1',5,1.0).
nn('X1',6,9.194521466621749e-17).
nn('X1',7,1.661802501162768e-10).
nn('X1',8,1.58666252337257e-13).
nn('X1',9,4.880273962726278e-11).
nn('X2',0,3.178878440386894e-15).
nn('X2',1,5.416791689673295e-13).
nn('X2',2,6.055785040093298e-11).
nn('X2',3,1.741421812523694e-13).
nn('X2',4,4.306711491743704e-15).
nn('X2',5,4.829741814815769e-11).
nn('X2',6,4.49901357488236e-12).
nn('X2',7,7.6369062185222e-09).
nn('X2',8,1.0).
nn('X2',9,2.9283965474746765e-08).
nn('X3',0,3.192308274820553e-14).
nn('X3',1,8.702377468812472e-13).
nn('X3',2,3.850261784204244e-14).
nn('X3',3,7.667905383106953e-17).
nn('X3',4,2.582165249170926e-21).
nn('X3',5,8.732518018157026e-14).
nn('X3',6,1.2819011009085883e-23).
nn('X3',7,1.0).
nn('X3',8,4.576207102300567e-22).
nn('X3',9,1.3236653526732534e-12).
nn('X4',0,9.491806849837303e-06).
nn('X4',1,3.6054292884557526e-10).
nn('X4',2,2.0023287916615118e-08).
nn('X4',3,2.8514358874076606e-08).
nn('X4',4,2.8389174691612062e-11).
nn('X4',5,0.9999831914901733).
nn('X4',6,3.268857199145714e-06).
nn('X4',7,2.658266566868406e-06).
nn('X4',8,1.1122860996692907e-06).
nn('X4',9,1.2528849424597865e-07).
nn('X5',0,3.0224167613113195e-10).
nn('X5',1,1.2490398715314655e-09).
nn('X5',2,5.34617940195492e-11).
nn('X5',3,1.9056193423239165e-07).
nn('X5',4,1.3726347532563854e-12).
nn('X5',5,0.9999997615814209).
nn('X5',6,9.396312937795526e-13).
nn('X5',7,1.7482184588857308e-08).
nn('X5',8,1.8464892700791047e-13).
nn('X5',9,2.4939567078163805e-10).

a :- Pos=[f(['X0','X1','X2','X3'],28),f(['X4','X5'],10)], metaabd(Pos).