:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,8.988543909140697e-20).
nn('X0',1,7.593272516030988e-17).
nn('X0',2,1.1217176005325763e-14).
nn('X0',3,1.215603952133687e-13).
nn('X0',4,1.6194356511171903e-13).
nn('X0',5,6.311236533385056e-10).
nn('X0',6,1.8462186875216867e-16).
nn('X0',7,1.637701956269666e-07).
nn('X0',8,0.9999984502792358).
nn('X0',9,1.3341757494345075e-06).
nn('X1',0,1.1005101896444103e-07).
nn('X1',1,2.41973260005357e-12).
nn('X1',2,2.6162402022578135e-08).
nn('X1',3,3.7473604948268235e-13).
nn('X1',4,5.691964233278668e-08).
nn('X1',5,3.190655115759e-05).
nn('X1',6,0.9999679923057556).
nn('X1',7,8.225027603447899e-11).
nn('X1',8,6.5359184731050846e-09).
nn('X1',9,4.789476541061655e-12).
nn('X2',0,2.7981264793197624e-05).
nn('X2',1,1.6238350326602813e-06).
nn('X2',2,0.9999703764915466).
nn('X2',3,6.848094091793655e-09).
nn('X2',4,5.936047128423131e-16).
nn('X2',5,2.6366539160327385e-12).
nn('X2',6,7.870168058921045e-12).
nn('X2',7,5.069463071549762e-09).
nn('X2',8,5.581056727876899e-10).
nn('X2',9,1.0879766055085782e-12).
nn('X3',0,7.933190808231079e-17).
nn('X3',1,8.327057555284083e-16).
nn('X3',2,2.861285963717819e-08).
nn('X3',3,1.1983076795335094e-26).
nn('X3',4,1.0).
nn('X3',5,4.540575204714514e-08).
nn('X3',6,8.786533722826562e-10).
nn('X3',7,4.418067948703912e-15).
nn('X3',8,3.1896950374748514e-19).
nn('X3',9,4.029088174206663e-10).
nn('X4',0,1.9494159175792447e-08).
nn('X4',1,1.0).
nn('X4',2,2.730104475823225e-10).
nn('X4',3,3.70994608321353e-20).
nn('X4',4,2.378055218021924e-12).
nn('X4',5,6.412063935368195e-11).
nn('X4',6,3.1221841084727586e-11).
nn('X4',7,4.709511114420195e-10).
nn('X4',8,5.111815918473761e-13).
nn('X4',9,9.065400340124707e-13).

a :- Pos=[f(['X0','X1','X2'],16),f(['X3','X4'],5)], metaabd(Pos).