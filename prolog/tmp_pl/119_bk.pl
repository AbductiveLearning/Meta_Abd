:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.00010142839892068878).
nn('X0',1,0.9998210072517395).
nn('X0',2,4.78210422443226e-05).
nn('X0',3,1.935769766703288e-08).
nn('X0',4,5.1725788097201075e-08).
nn('X0',5,1.992696343222633e-05).
nn('X0',6,2.2639969756710343e-06).
nn('X0',7,7.435383395204553e-06).
nn('X0',8,1.1829061286050546e-08).
nn('X0',9,2.475717231220642e-08).
nn('X1',0,2.266784946414191e-07).
nn('X1',1,0.9999996423721313).
nn('X1',2,1.1273749933593535e-08).
nn('X1',3,1.9866285598611739e-16).
nn('X1',4,4.0667810785599556e-10).
nn('X1',5,5.451691098556921e-09).
nn('X1',6,5.385269563618067e-09).
nn('X1',7,7.497187937133276e-08).
nn('X1',8,6.42147834906126e-10).
nn('X1',9,8.440133592024779e-10).
nn('X2',0,1.7689915239316178e-06).
nn('X2',1,1.2106582403248467e-07).
nn('X2',2,0.9999980330467224).
nn('X2',3,4.9978094850567506e-14).
nn('X2',4,1.7381942999776855e-19).
nn('X2',5,3.9758442759806176e-16).
nn('X2',6,1.2549451766392244e-13).
nn('X2',7,2.583057767679975e-09).
nn('X2',8,1.777497311665488e-12).
nn('X2',9,2.9276823495491587e-15).
nn('X3',0,1.0).
nn('X3',1,2.597000475182843e-15).
nn('X3',2,4.389631769186053e-09).
nn('X3',3,1.847564278767739e-16).
nn('X3',4,2.6450347319650425e-20).
nn('X3',5,1.0612862720799435e-13).
nn('X3',6,4.558943283647365e-13).
nn('X3',7,5.912626638293617e-11).
nn('X3',8,5.1682436948792196e-14).
nn('X3',9,4.0021120911639535e-14).

a :- Pos=[f(['X0','X1','X2','X3'],4)], metaabd(Pos).
