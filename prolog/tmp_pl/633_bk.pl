:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,4.7925020396633045e-08).
nn('X0',1,1.928709423282271e-07).
nn('X0',2,4.066178735229187e-05).
nn('X0',3,1.4861487215966918e-05).
nn('X0',4,2.5789056962821633e-05).
nn('X0',5,2.037982085312251e-05).
nn('X0',6,2.2788808564655483e-06).
nn('X0',7,0.0006564457435160875).
nn('X0',8,0.9903366565704346).
nn('X0',9,0.00890275463461876).
nn('X1',0,1.0).
nn('X1',1,1.4455698270693569e-15).
nn('X1',2,3.2105433867002375e-09).
nn('X1',3,2.1383367929258493e-16).
nn('X1',4,2.067615327700006e-20).
nn('X1',5,1.1359931550124214e-13).
nn('X1',6,1.989431883120396e-11).
nn('X1',7,3.5458404875482463e-12).
nn('X1',8,2.1290448245538107e-13).
nn('X1',9,6.8260737740141655e-15).
nn('X2',0,2.8679310662416135e-11).
nn('X2',1,1.7022779019715273e-11).
nn('X2',2,3.189928599045544e-10).
nn('X2',3,1.2871718354290351e-05).
nn('X2',4,1.2693461030721664e-05).
nn('X2',5,1.0820111384646225e-07).
nn('X2',6,3.126980011730618e-12).
nn('X2',7,0.005881079472601414).
nn('X2',8,1.3074935623080819e-06).
nn('X2',9,0.9940919280052185).
nn('X3',0,0.001960569294169545).
nn('X3',1,1.2726616982483563e-10).
nn('X3',2,1.5265111841245016e-08).
nn('X3',3,1.353216930871648e-11).
nn('X3',4,2.624510608839614e-09).
nn('X3',5,0.15179051458835602).
nn('X3',6,0.846248209476471).
nn('X3',7,1.803735216743263e-10).
nn('X3',8,7.588012067571981e-07).
nn('X3',9,1.7932659523989258e-10).
nn('X4',0,6.571029527793759e-13).
nn('X4',1,6.765799756520497e-15).
nn('X4',2,1.5856831906552182e-12).
nn('X4',3,7.866969298220283e-09).
nn('X4',4,1.6864677263583872e-06).
nn('X4',5,2.690530109816791e-08).
nn('X4',6,3.0705615611579086e-15).
nn('X4',7,8.314875594805926e-05).
nn('X4',8,1.7440811017621627e-09).
nn('X4',9,0.9999151825904846).

a :- Pos=[f(['X0','X1','X2','X3','X4'],32)], metaabd(Pos).
