:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,5.34524691175875e-09).
nn('X0',1,1.1322401904934054e-09).
nn('X0',2,1.0).
nn('X0',3,8.417146391367274e-17).
nn('X0',4,5.937240192522251e-26).
nn('X0',5,9.010480351244691e-20).
nn('X0',6,6.961575858047922e-17).
nn('X0',7,2.35548542476538e-11).
nn('X0',8,2.7531476108773843e-15).
nn('X0',9,2.1311346884966665e-19).
nn('X1',0,0.0010516446782276034).
nn('X1',1,1.78891718860541e-06).
nn('X1',2,0.9989372491836548).
nn('X1',3,5.269809548735793e-07).
nn('X1',4,6.602374354701401e-13).
nn('X1',5,1.2672509708266944e-09).
nn('X1',6,1.9451018573590773e-10).
nn('X1',7,8.903480193112046e-06).
nn('X1',8,9.50508116659421e-09).
nn('X1',9,1.6026949767322662e-09).
nn('X2',0,2.7881394567240425e-13).
nn('X2',1,6.950375275316389e-10).
nn('X2',2,3.6572853079341883e-10).
nn('X2',3,4.122174232179532e-06).
nn('X2',4,0.00023222151503432542).
nn('X2',5,0.00011078600073233247).
nn('X2',6,5.160257529439327e-13).
nn('X2',7,0.0002985755854751915).
nn('X2',8,3.2863408705452457e-07).
nn('X2',9,0.9993540048599243).
nn('X3',0,5.0102613613489666e-08).
nn('X3',1,3.8367244314940763e-07).
nn('X3',2,5.97954640397802e-05).
nn('X3',3,2.516933363949647e-06).
nn('X3',4,5.683348263119115e-07).
nn('X3',5,3.119879238511203e-06).
nn('X3',6,1.4291594197857194e-05).
nn('X3',7,0.00022262881975620985).
nn('X3',8,0.9989734888076782).
nn('X3',9,0.0007232192438095808).

a :- Pos=[f(['X0','X1','X2','X3'],21)], metaabd(Pos).
