:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,5.07334476651522e-08).
nn('X0',1,4.2741519687677226e-14).
nn('X0',2,1.2191597065225324e-08).
nn('X0',3,3.0126981994321256e-16).
nn('X0',4,8.115462719615607e-07).
nn('X0',5,1.0081683285534382e-05).
nn('X0',6,0.9999890327453613).
nn('X0',7,2.246661651205198e-13).
nn('X0',8,2.2637752783438714e-11).
nn('X0',9,2.4593375296325104e-13).
nn('X1',0,7.139649227383416e-08).
nn('X1',1,9.878489692916048e-11).
nn('X1',2,4.1266144279461514e-08).
nn('X1',3,1.0227420039338186e-12).
nn('X1',4,5.995031138184004e-09).
nn('X1',5,1.3236158338258974e-05).
nn('X1',6,0.9999866485595703).
nn('X1',7,5.119656087959612e-11).
nn('X1',8,5.042384731979155e-09).
nn('X1',9,2.2776626504988906e-13).
nn('X2',0,5.785401454780348e-13).
nn('X2',1,5.362075228276808e-09).
nn('X2',2,5.5254186008824036e-06).
nn('X2',3,2.922843824609833e-12).
nn('X2',4,0.9993953108787537).
nn('X2',5,0.00043958015157841146).
nn('X2',6,3.289678573992205e-08).
nn('X2',7,1.1562140116438968e-06).
nn('X2',8,1.0958848051245695e-08).
nn('X2',9,0.0001583525590831414).
nn('X3',0,1.5845826170299282e-12).
nn('X3',1,2.725947993807015e-13).
nn('X3',2,4.112729688060535e-12).
nn('X3',3,2.5069519438147836e-07).
nn('X3',4,1.2314915238675894e-06).
nn('X3',5,3.944400717159624e-08).
nn('X3',6,4.58405083098197e-15).
nn('X3',7,0.00020837492775171995).
nn('X3',8,4.9033747728799426e-08).
nn('X3',9,0.9997900724411011).
nn('X4',0,1.3390462072493392e-08).
nn('X4',1,1.0).
nn('X4',2,1.5036878409091514e-08).
nn('X4',3,1.8005217232384602e-17).
nn('X4',4,3.2093400853527854e-11).
nn('X4',5,3.2406138883445124e-10).
nn('X4',6,1.7179777045406297e-10).
nn('X4',7,3.097891720926782e-08).
nn('X4',8,1.3187652359025748e-10).
nn('X4',9,1.089749263050166e-10).
nn('X5',0,1.0968639685415837e-07).
nn('X5',1,1.3531763209950753e-10).
nn('X5',2,1.6467447197521246e-11).
nn('X5',3,4.3764378232502565e-10).
nn('X5',4,4.6478633441982e-11).
nn('X5',5,0.9999991059303284).
nn('X5',6,7.449340841958474e-07).
nn('X5',7,2.582829783381868e-10).
nn('X5',8,3.006594140320118e-10).
nn('X5',9,1.4595090414015743e-10).

a :- Pos=[f(['X0','X1','X2','X3'],25),f(['X4','X5'],6)], metaabd(Pos).