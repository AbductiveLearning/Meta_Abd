:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,3.0824239103006967e-13).
nn('X0',1,1.683384653894393e-09).
nn('X0',2,1.6215632285820902e-06).
nn('X0',3,0.9999982118606567).
nn('X0',4,4.517428407298502e-21).
nn('X0',5,9.432736192138691e-08).
nn('X0',6,3.815319280509339e-21).
nn('X0',7,7.893903087620446e-13).
nn('X0',8,6.54675867748687e-17).
nn('X0',9,2.3365382674186193e-16).
nn('X1',0,9.109414378372094e-08).
nn('X1',1,0.9999998807907104).
nn('X1',2,1.689883011302129e-09).
nn('X1',3,1.521915857158632e-18).
nn('X1',4,1.0913614456597998e-10).
nn('X1',5,3.400924541985262e-10).
nn('X1',6,1.0979659403886899e-09).
nn('X1',7,2.5627788780013816e-10).
nn('X1',8,1.5013976390268469e-12).
nn('X1',9,3.0852959076455022e-12).
nn('X2',0,4.063030023104125e-13).
nn('X2',1,3.190701591826439e-10).
nn('X2',2,5.019740265815864e-12).
nn('X2',3,1.0765499921367144e-14).
nn('X2',4,9.744000673056229e-18).
nn('X2',5,1.3768818713216086e-13).
nn('X2',6,6.810440667519628e-23).
nn('X2',7,1.0).
nn('X2',8,2.5445618333973543e-18).
nn('X2',9,3.0700131326000246e-10).

a :- Pos=[f(['X0','X1','X2'],11)], metaabd(Pos).
