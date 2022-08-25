:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,1.2618460106531008e-20).
nn('X0',2,1.092990919993042e-09).
nn('X0',3,6.829208307116625e-20).
nn('X0',4,6.1131197550512885e-28).
nn('X0',5,3.2234814880885212e-18).
nn('X0',6,1.4698914781772576e-16).
nn('X0',7,7.035667418131032e-15).
nn('X0',8,1.1701920432561841e-18).
nn('X0',9,5.999458264785854e-20).
nn('X1',0,0.0001573457702761516).
nn('X1',1,0.0015999242896214128).
nn('X1',2,0.9962987303733826).
nn('X1',3,1.0458171573191066e-06).
nn('X1',4,0.00011227849608985707).
nn('X1',5,0.0005683207418769598).
nn('X1',6,7.637511589564383e-05).
nn('X1',7,0.0006467651110142469).
nn('X1',8,0.00014705593639519066).
nn('X1',9,0.000392232759622857).
nn('X2',0,0.005887181963771582).
nn('X2',1,0.0010566844139248133).
nn('X2',2,0.3291100561618805).
nn('X2',3,0.6607365012168884).
nn('X2',4,3.51480848621577e-05).
nn('X2',5,0.000550468044821173).
nn('X2',6,7.362799078691751e-05).
nn('X2',7,0.00015844318841118366).
nn('X2',8,0.002287228824570775).
nn('X2',9,0.00010473415022715926).
nn('X3',0,2.8220670245104884e-08).
nn('X3',1,1.0).
nn('X3',2,1.4317709517985833e-10).
nn('X3',3,1.1023633364244124e-18).
nn('X3',4,4.9028780167714725e-12).
nn('X3',5,1.0919115611685015e-09).
nn('X3',6,6.620186343564427e-11).
nn('X3',7,3.1666822497555813e-09).
nn('X3',8,4.896184369398981e-13).
nn('X3',9,2.084627609422185e-12).
nn('X4',0,1.9638894677154184e-11).
nn('X4',1,3.0536981437180374e-20).
nn('X4',2,1.8970967104094821e-13).
nn('X4',3,1.8576646905511868e-22).
nn('X4',4,6.37334369307796e-13).
nn('X4',5,1.0198207434086726e-07).
nn('X4',6,0.9999998807907104).
nn('X4',7,5.759821146197098e-18).
nn('X4',8,7.660604091450519e-15).
nn('X4',9,2.40467013387694e-19).
nn('X5',0,4.981646029023068e-08).
nn('X5',1,1.5899453501333483e-05).
nn('X5',2,0.0004742745659314096).
nn('X5',3,7.270794100122657e-08).
nn('X5',4,0.9824054837226868).
nn('X5',5,0.004248664248734713).
nn('X5',6,8.122585768433055e-07).
nn('X5',7,0.0004379035672172904).
nn('X5',8,0.0003151136334054172).
nn('X5',9,0.012101679109036922).
nn('X6',0,9.183284688241145e-10).
nn('X6',1,3.943152023566654e-06).
nn('X6',2,4.655143857235089e-05).
nn('X6',3,0.9999343752861023).
nn('X6',4,1.0024765306065886e-12).
nn('X6',5,1.5113743756955955e-05).
nn('X6',6,1.443137974301966e-14).
nn('X6',7,8.383453931060103e-09).
nn('X6',8,5.417665274731753e-11).
nn('X6',9,6.84402673423179e-11).

a :- Pos=[f(['X0','X1','X2'],5),f(['X3','X4','X5','X6'],14)], metaabd(Pos).
