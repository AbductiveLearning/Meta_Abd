:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,1.0835703779516985e-22).
nn('X0',2,4.591070362422456e-14).
nn('X0',3,8.058145737801011e-26).
nn('X0',4,1.3585888995165363e-29).
nn('X0',5,2.822188710542711e-17).
nn('X0',6,6.005450043505833e-15).
nn('X0',7,2.232005848679905e-18).
nn('X0',8,6.465517671292372e-20).
nn('X0',9,9.00424296564445e-24).
nn('X1',0,7.34749969200088e-11).
nn('X1',1,9.35787447531311e-10).
nn('X1',2,2.0321413103374653e-06).
nn('X1',3,2.3838293874132478e-14).
nn('X1',4,0.999951958656311).
nn('X1',5,2.8295442461967468e-05).
nn('X1',6,7.994781299203169e-07).
nn('X1',7,8.749645452610366e-09).
nn('X1',8,9.110994077243806e-12).
nn('X1',9,1.6814019545563497e-05).
nn('X2',0,5.227617521086358e-07).
nn('X2',1,2.5311437639174983e-05).
nn('X2',2,0.00020190152281429619).
nn('X2',3,0.0004972357419319451).
nn('X2',4,0.3145434856414795).
nn('X2',5,0.008382364176213741).
nn('X2',6,5.4314223234541714e-05).
nn('X2',7,0.017736496403813362).
nn('X2',8,0.0012644474627450109).
nn('X2',9,0.6572938561439514).
nn('X3',0,5.162088978494239e-09).
nn('X3',1,1.3779514119960368e-05).
nn('X3',2,0.00015423195145558566).
nn('X3',3,0.999671459197998).
nn('X3',4,2.5068236375602737e-10).
nn('X3',5,0.00015867530601099133).
nn('X3',6,3.0499851776111253e-12).
nn('X3',7,1.8856172800951754e-06).
nn('X3',8,1.9776047466280033e-08).
nn('X3',9,2.6911664008366643e-08).
nn('X4',0,5.9696887433347e-09).
nn('X4',1,1.0).
nn('X4',2,1.309873459698352e-10).
nn('X4',3,8.554643733886592e-20).
nn('X4',4,4.4104948642917363e-13).
nn('X4',5,1.0866442468060455e-10).
nn('X4',6,4.817578974636305e-11).
nn('X4',7,2.5368235290201824e-10).
nn('X4',8,4.819706374455113e-13).
nn('X4',9,8.627328994433026e-13).
nn('X5',0,6.774961993326334e-13).
nn('X5',1,6.154504461619581e-08).
nn('X5',2,7.378685040748678e-07).
nn('X5',3,0.9999911785125732).
nn('X5',4,7.637220456539808e-18).
nn('X5',5,7.988017387106083e-06).
nn('X5',6,2.6410851254858608e-18).
nn('X5',7,9.45090533610582e-12).
nn('X5',8,2.3439224888945455e-15).
nn('X5',9,9.77564722421699e-14).
nn('X6',0,5.442935702149043e-08).
nn('X6',1,8.699112896692895e-08).
nn('X6',2,2.9190097006193128e-09).
nn('X6',3,4.0784758681411404e-08).
nn('X6',4,3.818475306616165e-06).
nn('X6',5,0.9967371821403503).
nn('X6',6,1.4994197954365518e-05).
nn('X6',7,1.467845322622452e-06).
nn('X6',8,0.0032262171152979136).
nn('X6',9,1.612751657376066e-05).

a :- Pos=[f(['X0','X1','X2'],13),f(['X3','X4','X5','X6'],12)], metaabd(Pos).
