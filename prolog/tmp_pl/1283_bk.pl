:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.8446582998876693e-06).
nn('X0',1,1.0074947603619377e-12).
nn('X0',2,1.176690833659677e-08).
nn('X0',3,1.5880453215996217e-14).
nn('X0',4,1.8082918984418939e-07).
nn('X0',5,0.00025988760171458125).
nn('X0',6,0.9997380375862122).
nn('X0',7,6.240995133882654e-13).
nn('X0',8,3.1710293058040406e-08).
nn('X0',9,3.182371683882468e-12).
nn('X1',0,2.1829862362210406e-06).
nn('X1',1,5.163004355643319e-13).
nn('X1',2,2.2218614503799472e-08).
nn('X1',3,2.9186500137161125e-15).
nn('X1',4,1.8847640603780746e-07).
nn('X1',5,9.314057388110086e-05).
nn('X1',6,0.9999045133590698).
nn('X1',7,2.1685534227697256e-13).
nn('X1',8,8.643351034898217e-10).
nn('X1',9,3.321902757209838e-13).
nn('X2',0,5.265717306635054e-10).
nn('X2',1,1.5602206193676693e-09).
nn('X2',2,2.46706488571391e-10).
nn('X2',3,1.1891877875314094e-05).
nn('X2',4,6.661214142127461e-14).
nn('X2',5,0.9999880194664001).
nn('X2',6,1.4448986169861289e-13).
nn('X2',7,5.4672725013915624e-08).
nn('X2',8,1.1455794503929719e-14).
nn('X2',9,3.2107210223841776e-09).
nn('X3',0,3.29338396831691e-11).
nn('X3',1,1.6250970838113665e-14).
nn('X3',2,3.865693598559261e-10).
nn('X3',3,4.716963486472991e-10).
nn('X3',4,0.0002710132976062596).
nn('X3',5,2.453013223657763e-07).
nn('X3',6,3.95676755249319e-11).
nn('X3',7,0.00024403959105256945).
nn('X3',8,3.1476518613793303e-10).
nn('X3',9,0.9994847178459167).
nn('X4',0,2.739980551474952e-13).
nn('X4',1,3.600134079739803e-12).
nn('X4',2,4.0208797713603417e-07).
nn('X4',3,9.987422084382382e-23).
nn('X4',4,0.9999980330467224).
nn('X4',5,1.4108983350524795e-06).
nn('X4',6,7.157947834457445e-08).
nn('X4',7,3.9220799459734057e-13).
nn('X4',8,5.771747618248623e-16).
nn('X4',9,7.572115379161914e-09).

a :- Pos=[f(['X0','X1','X2','X3','X4'],30)], metaabd(Pos).