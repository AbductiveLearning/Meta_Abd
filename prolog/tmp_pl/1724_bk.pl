:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0575883972827743e-13).
nn('X0',1,6.361207947110786e-14).
nn('X0',2,4.5130462951806227e-13).
nn('X0',3,6.536616581342969e-09).
nn('X0',4,8.107523171929643e-06).
nn('X0',5,6.307307387487526e-08).
nn('X0',6,4.473675979328512e-16).
nn('X0',7,5.5644632084295154e-05).
nn('X0',8,1.467299337587491e-10).
nn('X0',9,0.9999361038208008).
nn('X1',0,1.0734306670201477e-06).
nn('X1',1,0.999998927116394).
nn('X1',2,4.2028069913158106e-08).
nn('X1',3,2.917502019374787e-16).
nn('X1',4,1.1547522937860322e-09).
nn('X1',5,1.2360925616405893e-09).
nn('X1',6,2.4990673974656374e-09).
nn('X1',7,8.620849811791231e-09).
nn('X1',8,6.613257858001376e-11).
nn('X1',9,1.4691951821799165e-10).
nn('X2',0,4.673670446209144e-07).
nn('X2',1,2.916466167960513e-13).
nn('X2',2,1.2799160913345986e-06).
nn('X2',3,6.287778406803427e-15).
nn('X2',4,1.3344067156140227e-05).
nn('X2',5,4.4759579509445757e-07).
nn('X2',6,0.9999844431877136).
nn('X2',7,6.26549322407105e-13).
nn('X2',8,5.29484832714755e-12).
nn('X2',9,3.3588830501696254e-12).
nn('X3',0,0.00017126697639469057).
nn('X3',1,1.2377192153134597e-11).
nn('X3',2,8.366469739229387e-10).
nn('X3',3,1.3870055104447943e-10).
nn('X3',4,4.506362730888336e-13).
nn('X3',5,3.915854176739231e-05).
nn('X3',6,7.545500140503092e-13).
nn('X3',7,0.999789297580719).
nn('X3',8,3.266730852838351e-12).
nn('X3',9,2.4004228293961205e-07).
nn('X4',0,1.698962215357369e-08).
nn('X4',1,1.316124986772138e-09).
nn('X4',2,0.00013507500989362597).
nn('X4',3,1.0027876695250376e-13).
nn('X4',4,0.9995071887969971).
nn('X4',5,0.00011427285789977759).
nn('X4',6,0.00021185306832194328).
nn('X4',7,4.401766062755996e-08).
nn('X4',8,1.2717105146720087e-10).
nn('X4',9,3.149098120047711e-05).

a :- Pos=[f(['X0','X1','X2'],16),f(['X3','X4'],11)], metaabd(Pos).
