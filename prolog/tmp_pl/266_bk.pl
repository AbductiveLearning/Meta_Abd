:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,5.737426578573229e-27).
nn('X0',2,1.0606790583799052e-17).
nn('X0',3,4.961809985430977e-26).
nn('X0',4,7.852050707701805e-32).
nn('X0',5,1.0603085588895868e-14).
nn('X0',6,1.8499410002781753e-16).
nn('X0',7,3.6961386443250786e-16).
nn('X0',8,3.264676375063369e-19).
nn('X0',9,7.3970394036904e-22).
nn('X1',0,5.59462420834933e-10).
nn('X1',1,7.053550234559225e-06).
nn('X1',2,4.182660632068291e-05).
nn('X1',3,0.9990656971931458).
nn('X1',4,6.450531775392676e-11).
nn('X1',5,0.0008846662240102887).
nn('X1',6,1.7271557448128583e-12).
nn('X1',7,8.311592978316185e-07).
nn('X1',8,1.758997725254119e-09).
nn('X1',9,6.4785155018398655e-09).
nn('X2',0,3.472160337025798e-09).
nn('X2',1,6.06711910222657e-06).
nn('X2',2,2.0271096218493767e-05).
nn('X2',3,0.9999516010284424).
nn('X2',4,1.593830530159589e-16).
nn('X2',5,2.2119091227068566e-05).
nn('X2',6,2.60105200317615e-14).
nn('X2',7,1.1742559813399112e-08).
nn('X2',8,9.173576828724883e-12).
nn('X2',9,2.964689908499518e-12).
nn('X3',0,1.0).
nn('X3',1,8.63621891014632e-13).
nn('X3',2,4.7147736381703e-08).
nn('X3',3,2.3646039100325025e-16).
nn('X3',4,4.148590650697448e-13).
nn('X3',5,1.511302021028893e-11).
nn('X3',6,3.51572326806604e-09).
nn('X3',7,1.0856321267307956e-12).
nn('X3',8,3.8645963599398536e-13).
nn('X3',9,2.834169802058001e-13).
nn('X4',0,2.6615977157520733e-17).
nn('X4',1,1.861798435835776e-17).
nn('X4',2,1.0378064721952422e-15).
nn('X4',3,6.3532210603511885e-09).
nn('X4',4,1.2707176892945427e-06).
nn('X4',5,1.9440133058878928e-08).
nn('X4',6,5.279480776607582e-19).
nn('X4',7,0.0004656105302274227).
nn('X4',8,8.524752084793086e-13).
nn('X4',9,0.9995330572128296).

a :- Pos=[f(['X0','X1','X2','X3','X4'],15)], metaabd(Pos).
