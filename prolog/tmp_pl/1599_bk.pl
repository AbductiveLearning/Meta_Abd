:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,2.3423383233023287e-09).
nn('X0',1,6.940648750175527e-11).
nn('X0',2,1.8768817611203303e-08).
nn('X0',3,7.401632728942786e-07).
nn('X0',4,0.00018489010108169168).
nn('X0',5,7.842429477022961e-05).
nn('X0',6,7.338984281402006e-11).
nn('X0',7,0.000578062143176794).
nn('X0',8,2.3822924788419186e-07).
nn('X0',9,0.9991578459739685).
nn('X1',0,0.0002040939434664324).
nn('X1',1,6.22936249783379e-06).
nn('X1',2,0.0016271696658805013).
nn('X1',3,1.0016600754170213e-05).
nn('X1',4,0.08825766295194626).
nn('X1',5,0.6520857214927673).
nn('X1',6,0.2107311636209488).
nn('X1',7,7.106666453182697e-05).
nn('X1',8,0.04559650272130966).
nn('X1',9,0.0014103833818808198).
nn('X2',0,7.145768698979538e-11).
nn('X2',1,3.3044048183480945e-13).
nn('X2',2,2.3638251078761385e-11).
nn('X2',3,1.977296278099505e-11).
nn('X2',4,1.4960750860382177e-08).
nn('X2',5,0.9999998807907104).
nn('X2',6,2.6300279731827914e-09).
nn('X2',7,5.829151605851735e-10).
nn('X2',8,1.710562108625524e-10).
nn('X2',9,1.56294305497795e-07).
nn('X3',0,2.221755985703242e-15).
nn('X3',1,9.478588026230403e-14).
nn('X3',2,7.154330461389691e-08).
nn('X3',3,1.8824100265249578e-23).
nn('X3',4,0.9999995231628418).
nn('X3',5,3.1109823339647846e-07).
nn('X3',6,2.327620229891636e-08).
nn('X3',7,7.085745835563872e-13).
nn('X3',8,6.167213801734059e-17).
nn('X3',9,3.13519805672513e-08).

a :- Pos=[f(['X0','X1'],12),f(['X2','X3'],9)], metaabd(Pos).
