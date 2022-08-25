:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.9999995231628418).
nn('X0',1,3.398127836585986e-15).
nn('X0',2,4.999448606213264e-07).
nn('X0',3,7.8283372614349595e-16).
nn('X0',4,1.7281425381964095e-14).
nn('X0',5,1.0271430407604587e-11).
nn('X0',6,7.358474718444086e-09).
nn('X0',7,3.0767571650158754e-12).
nn('X0',8,6.228728643975501e-11).
nn('X0',9,6.211993809970839e-13).
nn('X1',0,9.339335065305931e-07).
nn('X1',1,1.9000737665919587e-05).
nn('X1',2,6.87789506628178e-05).
nn('X1',3,2.154998765035998e-05).
nn('X1',4,4.665481014853867e-07).
nn('X1',5,0.0001284381578443572).
nn('X1',6,0.0002561389119364321).
nn('X1',7,0.0003559872566256672).
nn('X1',8,0.9990912079811096).
nn('X1',9,5.757645703852177e-05).
nn('X2',0,6.513939236098276e-13).
nn('X2',1,1.7301884369658183e-09).
nn('X2',2,3.7078322634442884e-08).
nn('X2',3,1.9569636577898564e-08).
nn('X2',4,4.6626964511631286e-09).
nn('X2',5,1.0866541799714469e-07).
nn('X2',6,1.926864362511438e-10).
nn('X2',7,1.744533801684156e-05).
nn('X2',8,0.9999281764030457).
nn('X2',9,5.4282787459669635e-05).

a :- Pos=[f(['X0','X1','X2'],16)], metaabd(Pos).