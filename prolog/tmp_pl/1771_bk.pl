:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.9999911189079285).
nn('X0',1,7.455036588055464e-12).
nn('X0',2,6.0043548728572205e-06).
nn('X0',3,1.2768996970891067e-10).
nn('X0',4,2.309090418872728e-10).
nn('X0',5,1.714286526066644e-07).
nn('X0',6,1.405361672368599e-06).
nn('X0',7,3.657058655903711e-08).
nn('X0',8,1.2524042176664807e-06).
nn('X0',9,3.962883354802216e-08).
nn('X1',0,8.417630148471744e-09).
nn('X1',1,2.0563790760896053e-16).
nn('X1',2,2.1092105839670694e-10).
nn('X1',3,1.1378600553328393e-17).
nn('X1',4,4.432191280656639e-10).
nn('X1',5,4.013684815618035e-07).
nn('X1',6,0.9999996423721313).
nn('X1',7,4.579144138525115e-13).
nn('X1',8,6.796515607254694e-12).
nn('X1',9,3.4818603892022154e-14).
nn('X2',0,2.5448596563393266e-09).
nn('X2',1,9.355884600381614e-08).
nn('X2',2,6.500265953945927e-06).
nn('X2',3,5.3627427831770547e-08).
nn('X2',4,6.474052960392385e-10).
nn('X2',5,4.1019953300747147e-07).
nn('X2',6,7.033824545032985e-07).
nn('X2',7,2.601428423076868e-05).
nn('X2',8,0.9999628663063049).
nn('X2',9,3.4506854262872366e-06).

a :- Pos=[f(['X0','X1','X2'],14)], metaabd(Pos).
