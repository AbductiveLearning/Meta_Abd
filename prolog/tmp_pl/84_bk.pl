:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,1.9420514988656707e-17).
nn('X0',2,9.053197080888253e-10).
nn('X0',3,1.3964080943677214e-17).
nn('X0',4,1.324004727196771e-18).
nn('X0',5,2.5071177772728603e-12).
nn('X0',6,6.302271204905452e-11).
nn('X0',7,2.4225486525662754e-13).
nn('X0',8,2.7576979328564066e-12).
nn('X0',9,1.3322733046709345e-14).
nn('X1',0,4.2943176792320514e-11).
nn('X1',1,2.6962891297444003e-06).
nn('X1',2,1.5651439753128216e-06).
nn('X1',3,0.9998264312744141).
nn('X1',4,2.304387963000115e-13).
nn('X1',5,0.00016925969976000488).
nn('X1',6,3.269244362122956e-15).
nn('X1',7,5.331541874653567e-09).
nn('X1',8,2.2394325976948792e-12).
nn('X1',9,1.1367860125455209e-10).
nn('X2',0,2.176636542117194e-07).
nn('X2',1,5.293818335083689e-12).
nn('X2',2,1.1400256433313771e-10).
nn('X2',3,7.477972130409682e-10).
nn('X2',4,5.472456689192828e-12).
nn('X2',5,0.9999993443489075).
nn('X2',6,3.7398351082629233e-07).
nn('X2',7,2.0475849748374797e-12).
nn('X2',8,2.2438439452841796e-13).
nn('X2',9,3.151463248349251e-12).
nn('X3',0,7.742121830786597e-13).
nn('X3',1,4.103281950329496e-11).
nn('X3',2,2.9719998906330147e-08).
nn('X3',3,3.194423614516495e-09).
nn('X3',4,1.1767188612399337e-10).
nn('X3',5,2.0439163694874196e-09).
nn('X3',6,2.3240723581730016e-11).
nn('X3',7,6.460065833380213e-06).
nn('X3',8,0.9999575018882751).
nn('X3',9,3.61359998350963e-05).
nn('X4',0,2.1989274984202112e-11).
nn('X4',1,1.7295096910174834e-08).
nn('X4',2,1.7884412955027074e-06).
nn('X4',3,3.6980786433105095e-08).
nn('X4',4,1.6263083324474792e-08).
nn('X4',5,1.2793498171959072e-06).
nn('X4',6,8.46317789182649e-08).
nn('X4',7,2.2577216441277415e-05).
nn('X4',8,0.9999577403068542).
nn('X4',9,1.652958417253103e-05).
nn('X5',0,2.876616235880647e-06).
nn('X5',1,0.999997079372406).
nn('X5',2,1.121313033536353e-07).
nn('X5',3,5.290842650741744e-15).
nn('X5',4,2.2860291437609703e-08).
nn('X5',5,5.380741630034436e-09).
nn('X5',6,7.176631733329941e-09).
nn('X5',7,1.0472673395156562e-08).
nn('X5',8,6.17205730968351e-11).
nn('X5',9,3.412829185922561e-10).

a :- Pos=[f(['X0','X1'],3),f(['X2','X3','X4','X5'],22)], metaabd(Pos).
