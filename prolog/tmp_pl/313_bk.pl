:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,3.3048652703371317e-09).
nn('X0',1,3.965851647080854e-06).
nn('X0',2,5.042986595071852e-05).
nn('X0',3,0.9999028444290161).
nn('X0',4,9.783575452603291e-14).
nn('X0',5,4.2790736188180745e-05).
nn('X0',6,2.2917467077700204e-13).
nn('X0',7,3.893689370926268e-09).
nn('X0',8,3.983011837016548e-11).
nn('X0',9,5.199108851772216e-11).
nn('X1',0,4.4248494646126346e-07).
nn('X1',1,1.863681015068508e-10).
nn('X1',2,0.0004324676119722426).
nn('X1',3,1.8542195640792602e-19).
nn('X1',4,0.9971280694007874).
nn('X1',5,1.752218850015197e-05).
nn('X1',6,0.0024214163422584534).
nn('X1',7,2.160595660716247e-11).
nn('X1',8,1.3632191729617826e-11).
nn('X1',9,7.03692748516005e-08).
nn('X2',0,4.4071540985851296e-14).
nn('X2',1,2.254788356026438e-09).
nn('X2',2,1.2768637258631088e-09).
nn('X2',3,3.6386081647687973e-11).
nn('X2',4,5.838456900164601e-07).
nn('X2',5,2.06207189989982e-07).
nn('X2',6,1.7800673519289927e-14).
nn('X2',7,0.999987006187439).
nn('X2',8,4.0437735625287186e-12).
nn('X2',9,1.2246898222656455e-05).
nn('X3',0,1.362945455281317e-12).
nn('X3',1,1.6127074719055057e-12).
nn('X3',2,1.25948202062999e-13).
nn('X3',3,9.646713901865014e-14).
nn('X3',4,1.1653787608317465e-14).
nn('X3',5,6.213614622829766e-10).
nn('X3',6,3.1888678568622983e-19).
nn('X3',7,0.9999998807907104).
nn('X3',8,5.764216783972205e-18).
nn('X3',9,1.3504052276402945e-07).
nn('X4',0,1.4956079777417308e-11).
nn('X4',1,1.7650999409378303e-14).
nn('X4',2,5.135588135307678e-11).
nn('X4',3,1.7705052701710855e-10).
nn('X4',4,3.5791344998870045e-05).
nn('X4',5,2.2904221168573713e-06).
nn('X4',6,8.64709779474504e-13).
nn('X4',7,0.0010646507143974304).
nn('X4',8,7.485710717958227e-09).
nn('X4',9,0.9988971948623657).

a :- Pos=[f(['X0','X1','X2','X3','X4'],30)], metaabd(Pos).
