:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,7.765572417828436e-18).
nn('X0',2,8.272593721159183e-10).
nn('X0',3,8.418261554910087e-19).
nn('X0',4,4.712710482623856e-21).
nn('X0',5,3.790133625784092e-15).
nn('X0',6,1.6443199883295345e-13).
nn('X0',7,4.5436649776832427e-13).
nn('X0',8,8.131330624019245e-14).
nn('X0',9,1.5463920529019178e-15).
nn('X1',0,1.062762586268029e-10).
nn('X1',1,2.5372326035721926e-06).
nn('X1',2,9.153047722065821e-05).
nn('X1',3,0.9998428821563721).
nn('X1',4,1.1841576330606784e-10).
nn('X1',5,3.702851972775534e-05).
nn('X1',6,9.626660905058537e-14).
nn('X1',7,2.5847904908005148e-05).
nn('X1',8,5.565839700238939e-08).
nn('X1',9,1.0802233418871765e-07).
nn('X2',0,1.0).
nn('X2',1,5.52046963756422e-20).
nn('X2',2,1.1291464291351971e-11).
nn('X2',3,5.306369687979425e-21).
nn('X2',4,5.456523870816344e-22).
nn('X2',5,1.896132380339692e-15).
nn('X2',6,7.053531478513897e-13).
nn('X2',7,4.878409497350298e-16).
nn('X2',8,4.053643881609011e-15).
nn('X2',9,2.78387734416395e-17).
nn('X3',0,6.029950760932934e-09).
nn('X3',1,1.0).
nn('X3',2,1.1048426930282407e-11).
nn('X3',3,3.205683735935098e-21).
nn('X3',4,1.7705093305082215e-13).
nn('X3',5,1.253896708686142e-10).
nn('X3',6,3.888052380052187e-11).
nn('X3',7,2.397288444144774e-11).
nn('X3',8,3.8560236411246465e-14).
nn('X3',9,7.27523848998747e-14).

a :- Pos=[f(['X0','X1','X2','X3'],4)], metaabd(Pos).
