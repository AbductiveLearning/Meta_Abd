:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.9999963045120239).
nn('X0',1,2.2615853634277983e-09).
nn('X0',2,7.40385701192281e-07).
nn('X0',3,9.089843877596593e-10).
nn('X0',4,1.513717502851699e-14).
nn('X0',5,5.458194323182397e-07).
nn('X0',6,7.373342469918498e-08).
nn('X0',7,2.345178472751286e-06).
nn('X0',8,1.384368619739007e-09).
nn('X0',9,2.6117242049594758e-11).
nn('X1',0,2.1318552967386495e-07).
nn('X1',1,2.542111872116948e-07).
nn('X1',2,0.9999995231628418).
nn('X1',3,2.1425317235971263e-14).
nn('X1',4,5.08419019618532e-17).
nn('X1',5,1.2233854617080286e-15).
nn('X1',6,7.1567009046411e-13).
nn('X1',7,5.155936788625581e-10).
nn('X1',8,4.880129503628816e-13).
nn('X1',9,4.2847346925315995e-16).
nn('X2',0,4.3600263666121464e-07).
nn('X2',1,0.001205040025524795).
nn('X2',2,0.9953861832618713).
nn('X2',3,4.90096908833948e-07).
nn('X2',4,1.3272605237091284e-08).
nn('X2',5,2.647933250443657e-08).
nn('X2',6,3.769817453758151e-07).
nn('X2',7,0.0033944384194910526).
nn('X2',8,1.207727018481819e-05).
nn('X2',9,8.639861448500596e-07).
nn('X3',0,2.5271344045174615e-11).
nn('X3',1,4.015382210759144e-12).
nn('X3',2,1.4764494153496344e-05).
nn('X3',3,2.3220033654936546e-22).
nn('X3',4,0.9999821186065674).
nn('X3',5,2.9421917133731768e-06).
nn('X3',6,1.4589073771276162e-07).
nn('X3',7,2.2529666934140158e-12).
nn('X3',8,1.742769950162544e-14).
nn('X3',9,1.291127471603204e-08).
nn('X4',0,9.728737815350996e-10).
nn('X4',1,1.0).
nn('X4',2,2.4961159875158678e-11).
nn('X4',3,9.830335770563615e-22).
nn('X4',4,4.4956121209591485e-14).
nn('X4',5,6.061233329482385e-13).
nn('X4',6,1.8887194867985013e-14).
nn('X4',7,1.1731382532076395e-09).
nn('X4',8,4.0917383413788255e-15).
nn('X4',9,2.276519291545369e-14).

a :- Pos=[f(['X0','X1','X2'],4),f(['X3','X4'],5)], metaabd(Pos).