:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0763193358798162e-06).
nn('X0',1,6.061852408922874e-11).
nn('X0',2,6.500675226561725e-06).
nn('X0',3,1.4656132481683654e-13).
nn('X0',4,4.3533063944778405e-06).
nn('X0',5,4.112422539037652e-05).
nn('X0',6,0.9999468326568604).
nn('X0',7,3.248496263652534e-11).
nn('X0',8,5.103048650312303e-09).
nn('X0',9,4.7528779523187126e-11).
nn('X1',0,2.387344445395345e-12).
nn('X1',1,4.1523708083079924e-11).
nn('X1',2,7.116057965106393e-09).
nn('X1',3,7.240976152012024e-10).
nn('X1',4,3.043827967452728e-10).
nn('X1',5,1.031996488265463e-09).
nn('X1',6,3.364997724908392e-10).
nn('X1',7,1.1357230960129527e-06).
nn('X1',8,0.99998939037323).
nn('X1',9,9.491562195762526e-06).
nn('X2',0,2.5722266538963368e-09).
nn('X2',1,1.2242197792349067e-17).
nn('X2',2,5.2987201432097564e-11).
nn('X2',3,2.2694543441876206e-20).
nn('X2',4,3.9523609274283444e-08).
nn('X2',5,1.8027346868620953e-06).
nn('X2',6,0.9999982118606567).
nn('X2',7,3.058250837242074e-16).
nn('X2',8,1.8859944124005948e-14).
nn('X2',9,1.5188847638189528e-15).
nn('X3',0,2.8951114509206954e-13).
nn('X3',1,2.495565143423306e-10).
nn('X3',2,2.23112337494058e-08).
nn('X3',3,3.518844593486392e-08).
nn('X3',4,3.3346378991438996e-09).
nn('X3',5,5.689315685231122e-08).
nn('X3',6,6.281198478008676e-11).
nn('X3',7,3.283271144027822e-05).
nn('X3',8,0.9998935461044312).
nn('X3',9,7.345819904003292e-05).
nn('X4',0,3.386239314125128e-13).
nn('X4',1,2.688231376376729e-16).
nn('X4',2,7.005992465856925e-15).
nn('X4',3,8.336752295445837e-12).
nn('X4',4,2.9311898071046204e-16).
nn('X4',5,1.0).
nn('X4',6,2.1481919704451763e-13).
nn('X4',7,8.319113731225514e-10).
nn('X4',8,2.0569873929177085e-12).
nn('X4',9,1.0698034819123237e-10).
nn('X5',0,2.0943378975513127e-10).
nn('X5',1,6.69904132166721e-11).
nn('X5',2,1.575498842498746e-09).
nn('X5',3,1.2101855628721125e-10).
nn('X5',4,6.7403842923807744e-12).
nn('X5',5,2.1066585986773134e-07).
nn('X5',6,1.6663850033182825e-07).
nn('X5',7,1.3348387284395358e-08).
nn('X5',8,0.9999996423721313).
nn('X5',9,6.237690541865959e-08).

a :- Pos=[f(['X0','X1'],14),f(['X2','X3','X4','X5'],27)], metaabd(Pos).
