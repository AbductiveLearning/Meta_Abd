:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.1994613080545946e-09).
nn('X0',1,1.0323047717974987e-05).
nn('X0',2,4.231008279020898e-05).
nn('X0',3,0.999809980392456).
nn('X0',4,1.4661551139827367e-10).
nn('X0',5,0.00013737181143369526).
nn('X0',6,3.3578372829641545e-13).
nn('X0',7,2.8208667401941057e-08).
nn('X0',8,3.101678636152627e-10).
nn('X0',9,1.7194209389614912e-09).
nn('X1',0,4.921388153888984e-06).
nn('X1',1,4.0873335271918165e-10).
nn('X1',2,7.890039199764942e-08).
nn('X1',3,1.4903006606559188e-10).
nn('X1',4,1.0786950497276848e-06).
nn('X1',5,0.00144576549064368).
nn('X1',6,0.9985479116439819).
nn('X1',7,2.3018928985152343e-09).
nn('X1',8,1.705126493334319e-07).
nn('X1',9,3.2745822720059437e-10).
nn('X2',0,2.6738720503384006e-16).
nn('X2',1,4.335952392571903e-16).
nn('X2',2,7.162800115196757e-18).
nn('X2',3,3.782456283345531e-19).
nn('X2',4,2.343595511843029e-19).
nn('X2',5,4.854670742552962e-14).
nn('X2',6,1.8864336510150828e-25).
nn('X2',7,1.0).
nn('X2',8,9.522501703408404e-24).
nn('X2',9,2.9930963263424815e-10).

a :- Pos=[f(['X0','X1','X2'],16)], metaabd(Pos).
