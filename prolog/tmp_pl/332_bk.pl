:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.9999986886978149).
nn('X0',1,2.6620129643900418e-14).
nn('X0',2,1.240831807081122e-06).
nn('X0',3,1.4132310620552533e-15).
nn('X0',4,1.2200771839501138e-14).
nn('X0',5,4.105967649215003e-11).
nn('X0',6,1.4764950151402445e-08).
nn('X0',7,3.4878911282613945e-13).
nn('X0',8,1.2993992166976387e-11).
nn('X0',9,1.4822220057233992e-13).
nn('X1',0,2.1713908449072505e-11).
nn('X1',1,5.381100296256158e-13).
nn('X1',2,1.7815592479996506e-13).
nn('X1',3,4.968476671751709e-11).
nn('X1',4,5.2475693468291215e-14).
nn('X1',5,1.0).
nn('X1',6,4.4643025387536284e-11).
nn('X1',7,3.2029114603593367e-12).
nn('X1',8,1.603726049528853e-12).
nn('X1',9,2.727675941019414e-12).
nn('X2',0,1.0735757985727568e-12).
nn('X2',1,1.153345641213832e-09).
nn('X2',2,6.355518195055865e-11).
nn('X2',3,4.293736199922904e-11).
nn('X2',4,6.226477124691987e-13).
nn('X2',5,7.737283903852532e-12).
nn('X2',6,5.904350588918156e-20).
nn('X2',7,0.9999974966049194).
nn('X2',8,2.5753560742303172e-11).
nn('X2',9,2.526189064155915e-06).
nn('X3',0,3.754021093982374e-09).
nn('X3',1,1.8702867237152532e-06).
nn('X3',2,3.641497460193932e-05).
nn('X3',3,3.390832353034057e-05).
nn('X3',4,0.00010430064867250621).
nn('X3',5,4.5229229726828635e-05).
nn('X3',6,5.206457043271939e-09).
nn('X3',7,0.9968770742416382).
nn('X3',8,1.3901566831009404e-07).
nn('X3',9,0.002901150146499276).
nn('X4',0,7.625367715036191e-08).
nn('X4',1,6.132282123871846e-06).
nn('X4',2,4.9135254812426865e-05).
nn('X4',3,1.5945547602314036e-06).
nn('X4',4,2.3269500104561303e-08).
nn('X4',5,4.711953442892991e-05).
nn('X4',6,0.00036114759859628975).
nn('X4',7,3.346961602801457e-05).
nn('X4',8,0.9995011687278748).
nn('X4',9,2.4327727032869007e-07).
nn('X5',0,2.2762481677318647e-08).
nn('X5',1,0.9999995231628418).
nn('X5',2,6.046835032691433e-09).
nn('X5',3,9.117124492810877e-17).
nn('X5',4,2.467695492391897e-10).
nn('X5',5,1.023028328717146e-09).
nn('X5',6,1.1111282727238603e-10).
nn('X5',7,4.5690853767155204e-07).
nn('X5',8,1.521100745360826e-10).
nn('X5',9,4.908778383772017e-10).

a :- Pos=[f(['X0','X1','X2'],12),f(['X3','X4','X5'],16)], metaabd(Pos).
