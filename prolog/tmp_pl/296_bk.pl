:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.0001247223117388785).
nn('X0',1,3.435566395637579e-05).
nn('X0',2,0.9998379945755005).
nn('X0',3,1.0945155963781872e-06).
nn('X0',4,1.0109380238314258e-13).
nn('X0',5,9.831726544007324e-10).
nn('X0',6,5.002269176657137e-08).
nn('X0',7,1.6139540548465448e-06).
nn('X0',8,2.2716129421951337e-07).
nn('X0',9,2.6428847998971605e-10).
nn('X1',0,0.0001102328096749261).
nn('X1',1,2.7782909306495185e-10).
nn('X1',2,6.574966793237991e-09).
nn('X1',3,1.7869883090632088e-12).
nn('X1',4,2.826267575528618e-07).
nn('X1',5,0.01922265626490116).
nn('X1',6,0.98066645860672).
nn('X1',7,1.8537260615403284e-10).
nn('X1',8,4.5360673084360315e-07).
nn('X1',9,3.1023839053290203e-09).
nn('X2',0,9.864979944040897e-09).
nn('X2',1,2.8516140559986525e-09).
nn('X2',2,2.663453457785181e-08).
nn('X2',3,1.6172782125067897e-05).
nn('X2',4,0.0055388775654137135).
nn('X2',5,0.00017469872545916587).
nn('X2',6,7.66148922082266e-09).
nn('X2',7,0.001683046342805028).
nn('X2',8,6.003189696457412e-07).
nn('X2',9,0.9925864934921265).
nn('X3',0,2.8325302992016077e-07).
nn('X3',1,4.2392569218462767e-14).
nn('X3',2,6.65881794148504e-09).
nn('X3',3,9.141467425621578e-17).
nn('X3',4,1.852302739280276e-05).
nn('X3',5,2.4860839403118007e-05).
nn('X3',6,0.9999562501907349).
nn('X3',7,2.6329172132452672e-14).
nn('X3',8,1.7777007079930462e-12).
nn('X3',9,5.238441516501602e-13).
nn('X4',0,1.5888461657453945e-09).
nn('X4',1,1.7605571536874507e-18).
nn('X4',2,1.9966946390552465e-13).
nn('X4',3,9.994973187184902e-22).
nn('X4',4,1.9986016314144095e-11).
nn('X4',5,2.6454017643118277e-06).
nn('X4',6,0.9999973773956299).
nn('X4',7,1.5037547966173005e-18).
nn('X4',8,1.1630383240951881e-14).
nn('X4',9,2.726106057566511e-19).
nn('X5',0,9.648531973383001e-11).
nn('X5',1,1.6527968682567007e-06).
nn('X5',2,1.095632524084067e-05).
nn('X5',3,0.9999624490737915).
nn('X5',4,3.794005243979402e-14).
nn('X5',5,2.4959497750387527e-05).
nn('X5',6,2.937896931618335e-15).
nn('X5',7,7.03657132561375e-10).
nn('X5',8,2.782281132915343e-12).
nn('X5',9,2.144913587021069e-12).
nn('X6',0,3.2545027870561904e-13).
nn('X6',1,4.595648514515993e-12).
nn('X6',2,1.4640480494598762e-10).
nn('X6',3,3.702291806462199e-08).
nn('X6',4,1.1819921708067227e-09).
nn('X6',5,0.999958336353302).
nn('X6',6,3.5629013628302175e-10).
nn('X6',7,1.4565461015081382e-06).
nn('X6',8,3.0239269108278677e-05).
nn('X6',9,9.885065082926303e-06).

a :- Pos=[f(['X0','X1','X2','X3','X4'],29),f(['X5','X6'],8)], metaabd(Pos).
