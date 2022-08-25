:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,9.927628497052865e-09).
nn('X0',1,1.9113154525366703e-17).
nn('X0',2,3.4557892571868143e-12).
nn('X0',3,3.7078096307876704e-20).
nn('X0',4,1.6519999812558694e-09).
nn('X0',5,1.0597927030175924e-05).
nn('X0',6,0.99998939037323).
nn('X0',7,8.25220326850175e-17).
nn('X0',8,2.2602189217457713e-12).
nn('X0',9,2.3457827850049528e-15).
nn('X1',0,0.00010158832446904853).
nn('X1',1,0.0007014452712610364).
nn('X1',2,0.0001345088821835816).
nn('X1',3,4.059208004036918e-05).
nn('X1',4,7.679925033698964e-07).
nn('X1',5,0.00015881715808063745).
nn('X1',6,0.0015019724378362298).
nn('X1',7,0.0003886084596160799).
nn('X1',8,0.996696949005127).
nn('X1',9,0.0002748994156718254).
nn('X2',0,4.400242687552236e-06).
nn('X2',1,5.086767123430036e-05).
nn('X2',2,0.9999220371246338).
nn('X2',3,7.888790065635476e-08).
nn('X2',4,1.381938175004649e-10).
nn('X2',5,7.006001112586091e-09).
nn('X2',6,9.64128048508428e-08).
nn('X2',7,1.9409948436077684e-05).
nn('X2',8,3.097493390669115e-06).
nn('X2',9,1.579591746292408e-08).
nn('X3',0,2.128352702412961e-11).
nn('X3',1,6.328244039899289e-10).
nn('X3',2,4.495004759519361e-06).
nn('X3',3,1.6346931200708105e-15).
nn('X3',4,0.9999575614929199).
nn('X3',5,3.1516578019363806e-05).
nn('X3',6,5.579340722761117e-07).
nn('X3',7,5.291915239347134e-10).
nn('X3',8,2.565129879339434e-11).
nn('X3',9,5.952545507170726e-06).
nn('X4',0,7.049789640944937e-10).
nn('X4',1,1.0).
nn('X4',2,2.2200448077103196e-11).
nn('X4',3,2.8598822888162486e-24).
nn('X4',4,2.8923833949668146e-14).
nn('X4',5,1.4272808737843257e-13).
nn('X4',6,8.716244143548019e-14).
nn('X4',7,4.4944484332148926e-13).
nn('X4',8,1.0972675498177836e-15).
nn('X4',9,1.1330230339828742e-15).
nn('X5',0,1.5392350505294417e-08).
nn('X5',1,1.0).
nn('X5',2,7.405692031836608e-11).
nn('X5',3,7.2937410484051e-19).
nn('X5',4,1.1762470338017028e-11).
nn('X5',5,3.645793111850537e-10).
nn('X5',6,7.1443480471888865e-12).
nn('X5',7,9.737301631673745e-09).
nn('X5',8,2.9067641848200265e-13).
nn('X5',9,4.16678453893371e-12).

a :- Pos=[f(['X0','X1','X2'],16),f(['X3','X4','X5'],6)], metaabd(Pos).