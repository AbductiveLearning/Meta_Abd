:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.1815819433991237e-09).
nn('X0',1,2.183779542974662e-06).
nn('X0',2,0.00032465136609971523).
nn('X0',3,0.0014252709224820137).
nn('X0',4,0.0024510386865586042).
nn('X0',5,4.967985296389088e-05).
nn('X0',6,4.938020548017619e-10).
nn('X0',7,0.9735841155052185).
nn('X0',8,7.02156285115052e-06).
nn('X0',9,0.022156188264489174).
nn('X1',0,4.698202701547416e-06).
nn('X1',1,0.0003027549828402698).
nn('X1',2,0.9995500445365906).
nn('X1',3,1.9662950307974825e-06).
nn('X1',4,1.6593912008366907e-11).
nn('X1',5,1.2534191462520994e-09).
nn('X1',6,2.0032932257496583e-10).
nn('X1',7,0.00014013667532708496).
nn('X1',8,3.2170748909265967e-07).
nn('X1',9,8.858222599883447e-09).
nn('X2',0,7.704249682660702e-09).
nn('X2',1,8.566300557788509e-09).
nn('X2',2,6.017617488396354e-05).
nn('X2',3,1.4623420530046616e-11).
nn('X2',4,0.9994714856147766).
nn('X2',5,0.0002372838498558849).
nn('X2',6,4.932450974592939e-05).
nn('X2',7,1.5095305627710331e-07).
nn('X2',8,1.5574347367319774e-09).
nn('X2',9,0.00018159417959395796).
nn('X3',0,1.0).
nn('X3',1,4.399446114646983e-17).
nn('X3',2,2.267130705391196e-09).
nn('X3',3,1.3837346120486127e-17).
nn('X3',4,3.897226559361498e-20).
nn('X3',5,7.546863660260264e-14).
nn('X3',6,7.164420966210283e-12).
nn('X3',7,4.5038536160106613e-13).
nn('X3',8,1.113350405791691e-12).
nn('X3',9,9.374839534499139e-15).
nn('X4',0,4.1997811207460795e-14).
nn('X4',1,1.499198511102813e-14).
nn('X4',2,1.137619322745878e-13).
nn('X4',3,9.676560353000241e-08).
nn('X4',4,7.587197359271158e-08).
nn('X4',5,4.796855179733939e-08).
nn('X4',6,4.937814631093651e-17).
nn('X4',7,0.0005673011182807386).
nn('X4',8,7.517535038914502e-08).
nn('X4',9,0.9994325041770935).
nn('X5',0,5.8280232195784265e-08).
nn('X5',1,1.1715164462708458e-14).
nn('X5',2,1.2846697663992934e-11).
nn('X5',3,2.405233433534331e-16).
nn('X5',4,9.645331061625129e-11).
nn('X5',5,0.0005864615668542683).
nn('X5',6,0.9994136095046997).
nn('X5',7,1.0056204865138349e-13).
nn('X5',8,7.542403368532291e-10).
nn('X5',9,6.746058383135314e-15).

a :- Pos=[f(['X0','X1','X2','X3'],13),f(['X4','X5'],15)], metaabd(Pos).
