:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,1.229739660796314e-11).
nn('X0',2,1.6553032722299577e-08).
nn('X0',3,3.882499790414107e-12).
nn('X0',4,2.9964815418987052e-15).
nn('X0',5,9.465667832886382e-11).
nn('X0',6,2.2177648606458433e-10).
nn('X0',7,1.6398139734974393e-08).
nn('X0',8,1.6293680293877344e-10).
nn('X0',9,3.0779637172351215e-10).
nn('X1',0,0.05101607367396355).
nn('X1',1,1.5241015717037953e-05).
nn('X1',2,0.9249991178512573).
nn('X1',3,3.3629157769610174e-06).
nn('X1',4,0.005286549683660269).
nn('X1',5,0.00011824416287709028).
nn('X1',6,0.018240919336676598).
nn('X1',7,4.7730395635880996e-06).
nn('X1',8,0.00016824179328978062).
nn('X1',9,0.00014754295989405364).
nn('X2',0,1.4368160327270593e-17).
nn('X2',1,2.536115535175967e-15).
nn('X2',2,1.5736682790201684e-11).
nn('X2',3,2.6857229412429948e-14).
nn('X2',4,3.698723185369124e-14).
nn('X2',5,3.2444806932242776e-13).
nn('X2',6,2.870913566149465e-15).
nn('X2',7,4.4400650267562014e-08).
nn('X2',8,0.9999996423721313).
nn('X2',9,3.367606211668317e-07).
nn('X3',0,3.1890892704389273e-10).
nn('X3',1,3.2384950827690773e-06).
nn('X3',2,3.024143006769009e-05).
nn('X3',3,0.9997934699058533).
nn('X3',4,5.716863094029634e-10).
nn('X3',5,0.00016620974929537624).
nn('X3',6,2.5078528663458055e-13).
nn('X3',7,6.823208877904108e-06).
nn('X3',8,5.807262581924988e-08).
nn('X3',9,7.805140711525382e-08).
nn('X4',0,6.6061138568329625e-06).
nn('X4',1,0.9999513626098633).
nn('X4',2,3.95131173718255e-05).
nn('X4',3,2.0616350640545456e-11).
nn('X4',4,2.093465582220233e-06).
nn('X4',5,1.5235936956514706e-08).
nn('X4',6,2.7818888526098817e-08).
nn('X4',7,4.9571912796864126e-08).
nn('X4',8,3.755711475150747e-07).
nn('X4',9,9.603156492232756e-09).

a :- Pos=[f(['X0','X1','X2','X3','X4'],14)], metaabd(Pos).