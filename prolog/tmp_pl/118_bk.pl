:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,2.197827640368355e-12).
nn('X0',1,6.657401019349862e-11).
nn('X0',2,3.909874067176133e-06).
nn('X0',3,5.520526222301588e-15).
nn('X0',4,0.9999517202377319).
nn('X0',5,1.1542964784894139e-05).
nn('X0',6,6.116925987953437e-08).
nn('X0',7,1.477076096989549e-07).
nn('X0',8,5.850171692564632e-13).
nn('X0',9,3.266982457716949e-05).
nn('X1',0,1.4426484540308593e-06).
nn('X1',1,3.6350693335407414e-06).
nn('X1',2,2.3639956907572923e-06).
nn('X1',3,0.002690209774300456).
nn('X1',4,0.0022383315954357386).
nn('X1',5,0.8955987095832825).
nn('X1',6,1.2287932804611046e-06).
nn('X1',7,0.05038129538297653).
nn('X1',8,0.00017168218619190156).
nn('X1',9,0.04891113191843033).
nn('X2',0,6.674692742958399e-11).
nn('X2',1,5.7438955813893244e-09).
nn('X2',2,1.0966615116014822e-10).
nn('X2',3,1.444148689522251e-10).
nn('X2',4,1.4706074899506483e-11).
nn('X2',5,6.582524036957693e-08).
nn('X2',6,4.0437268198625573e-16).
nn('X2',7,0.9999979734420776).
nn('X2',8,6.352284285604873e-14).
nn('X2',9,1.979320131795248e-06).
nn('X3',0,1.0).
nn('X3',1,1.0120678871678177e-18).
nn('X3',2,1.0282348757162385e-11).
nn('X3',3,4.28599878154265e-21).
nn('X3',4,1.5517802181814438e-24).
nn('X3',5,1.3071141126787445e-16).
nn('X3',6,2.994049371493959e-14).
nn('X3',7,2.1033880017639e-15).
nn('X3',8,3.972796810255074e-16).
nn('X3',9,8.13315617920567e-19).

a :- Pos=[f(['X0','X1','X2','X3'],16)], metaabd(Pos).
