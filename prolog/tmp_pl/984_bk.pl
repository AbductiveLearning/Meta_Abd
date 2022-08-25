:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,3.132700109926721e-15).
nn('X0',2,2.6552653409339655e-09).
nn('X0',3,3.578230715743126e-14).
nn('X0',4,4.805798590563658e-17).
nn('X0',5,2.2421177761633437e-11).
nn('X0',6,2.0302183545428676e-11).
nn('X0',7,1.737143406899122e-08).
nn('X0',8,7.137242186150417e-11).
nn('X0',9,4.341831061349666e-11).
nn('X1',0,4.937285580375317e-10).
nn('X1',1,1.0).
nn('X1',2,3.5825658412091954e-11).
nn('X1',3,8.827227311678287e-23).
nn('X1',4,3.74330795039398e-14).
nn('X1',5,1.3815163273892157e-14).
nn('X1',6,1.3278862457513966e-15).
nn('X1',7,1.3653118913214968e-10).
nn('X1',8,5.196530614262661e-15).
nn('X1',9,2.848474181069041e-15).
nn('X2',0,1.644244046472565e-10).
nn('X2',1,6.3300223018814e-07).
nn('X2',2,7.403338258882286e-06).
nn('X2',3,0.9999704360961914).
nn('X2',4,2.010589062590145e-14).
nn('X2',5,2.1525751435547136e-05).
nn('X2',6,4.488016776521119e-16).
nn('X2',7,2.985302560709613e-10).
nn('X2',8,1.6474466918697112e-13).
nn('X2',9,1.3800487662363192e-11).
nn('X3',0,5.636871720261782e-13).
nn('X3',1,6.614208969867914e-17).
nn('X3',2,5.209854357819677e-13).
nn('X3',3,4.046917800870631e-10).
nn('X3',4,1.9072821544341423e-08).
nn('X3',5,2.545474941939574e-10).
nn('X3',6,1.673643824271181e-15).
nn('X3',7,6.946490611881018e-05).
nn('X3',8,3.8354095011072786e-08).
nn('X3',9,0.9999305605888367).
nn('X4',0,2.291429268552747e-08).
nn('X4',1,9.489111055347322e-15).
nn('X4',2,2.0324105065583353e-08).
nn('X4',3,2.7689015700459222e-17).
nn('X4',4,6.814127484489063e-09).
nn('X4',5,1.2174756193417124e-05).
nn('X4',6,0.9999879002571106).
nn('X4',7,4.0195673226244965e-15).
nn('X4',8,1.7072349089986005e-09).
nn('X4',9,6.325600714887289e-14).
nn('X5',0,7.363434662011059e-08).
nn('X5',1,5.980319201626116e-06).
nn('X5',2,0.9999938607215881).
nn('X5',3,2.8442816678297955e-12).
nn('X5',4,2.860894931586874e-18).
nn('X5',5,5.371086740516354e-15).
nn('X5',6,7.274947110653615e-14).
nn('X5',7,4.1379166759725194e-08).
nn('X5',8,1.465461085814468e-11).
nn('X5',9,3.951532789268084e-15).
nn('X6',0,1.5606030356885014e-11).
nn('X6',1,1.9790366567740136e-10).
nn('X6',2,1.0).
nn('X6',3,6.609766182797892e-18).
nn('X6',4,4.779809528649185e-28).
nn('X6',5,1.2292943179448966e-22).
nn('X6',6,3.4377270828534374e-21).
nn('X6',7,5.470888499170545e-13).
nn('X6',8,1.8670979173226072e-16).
nn('X6',9,1.7360910014192972e-22).

a :- Pos=[f(['X0','X1','X2','X3'],13),f(['X4','X5','X6'],10)], metaabd(Pos).