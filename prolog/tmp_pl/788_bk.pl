:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,9.120742249937939e-17).
nn('X0',2,7.278806446464614e-10).
nn('X0',3,4.948713893716895e-17).
nn('X0',4,8.985312007858701e-22).
nn('X0',5,4.3849594636748146e-14).
nn('X0',6,1.981512393403606e-12).
nn('X0',7,1.722990240066169e-12).
nn('X0',8,4.3319728094395535e-14).
nn('X0',9,6.901954373560995e-15).
nn('X1',0,0.0012821127893403172).
nn('X1',1,7.40562938972289e-07).
nn('X1',2,0.9986670613288879).
nn('X1',3,2.3906657588668168e-05).
nn('X1',4,5.0182396432729703e-11).
nn('X1',5,8.461430667239256e-09).
nn('X1',6,7.346837804789175e-08).
nn('X1',7,2.533392034820281e-05).
nn('X1',8,4.7389968926836445e-07).
nn('X1',9,3.1919938692226424e-07).
nn('X2',0,1.1103367123285458e-16).
nn('X2',1,1.0649726949916657e-13).
nn('X2',2,5.770329093018889e-12).
nn('X2',3,5.1706447283528245e-12).
nn('X2',4,1.949702118503316e-12).
nn('X2',5,4.111441409548888e-09).
nn('X2',6,5.796388759596316e-14).
nn('X2',7,7.31224872652092e-07).
nn('X2',8,0.9999978542327881).
nn('X2',9,1.413311679243634e-06).
nn('X3',0,0.9999951720237732).
nn('X3',1,2.006945581944919e-10).
nn('X3',2,2.2761950901895034e-07).
nn('X3',3,7.07079395034782e-11).
nn('X3',4,3.3121107852890352e-12).
nn('X3',5,1.4614424799219705e-06).
nn('X3',6,2.803828238029382e-06).
nn('X3',7,4.501681871715846e-07).
nn('X3',8,8.139552226671398e-11).
nn('X3',9,3.240970686269051e-11).

a :- Pos=[f(['X0','X1','X2','X3'],10)], metaabd(Pos).
