:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,1.0643583034608134e-19).
nn('X0',2,7.79152621328133e-13).
nn('X0',3,1.5774819967268364e-22).
nn('X0',4,2.3179588440781503e-22).
nn('X0',5,2.97516934591284e-16).
nn('X0',6,2.2558969191917473e-14).
nn('X0',7,1.1332262160110968e-15).
nn('X0',8,4.8800053769060964e-18).
nn('X0',9,7.452397702012797e-18).
nn('X1',0,8.76529141695219e-14).
nn('X1',1,1.31272067669618e-16).
nn('X1',2,1.4397130135204023e-13).
nn('X1',3,7.03719216232912e-09).
nn('X1',4,5.555914981414389e-07).
nn('X1',5,2.4212598592754375e-09).
nn('X1',6,6.03433286115558e-17).
nn('X1',7,0.00023634367971681058).
nn('X1',8,7.079481098570817e-12).
nn('X1',9,0.9997630715370178).
nn('X2',0,3.772691172798659e-07).
nn('X2',1,4.8395220801467076e-05).
nn('X2',2,0.0011144491145387292).
nn('X2',3,0.9986149668693542).
nn('X2',4,7.848320326964142e-10).
nn('X2',5,0.00022095488384366035).
nn('X2',6,1.8523299560868622e-10).
nn('X2',7,6.766337037333869e-07).
nn('X2',8,1.4214916177479608e-07).
nn('X2',9,4.189462998738236e-09).
nn('X3',0,6.252108830651082e-10).
nn('X3',1,1.2045572872487753e-10).
nn('X3',2,1.2455800280086748e-10).
nn('X3',3,2.043562830067458e-08).
nn('X3',4,7.249408864005991e-13).
nn('X3',5,1.0).
nn('X3',6,2.5001578585204243e-10).
nn('X3',7,3.345509286267756e-11).
nn('X3',8,1.7157877495090762e-13).
nn('X3',9,1.3865288257808306e-11).
nn('X4',0,1.8687220659785453e-09).
nn('X4',1,4.321650439780861e-17).
nn('X4',2,1.595882453964137e-10).
nn('X4',3,1.5992065796198211e-18).
nn('X4',4,9.138784839990421e-09).
nn('X4',5,2.056991888821358e-06).
nn('X4',6,0.9999979734420776).
nn('X4',7,5.244744576653323e-15).
nn('X4',8,2.4732295204887844e-12).
nn('X4',9,3.773298845957413e-15).

a :- Pos=[f(['X0','X1','X2','X3','X4'],23)], metaabd(Pos).
