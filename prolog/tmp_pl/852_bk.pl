:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.0002569021889939904).
nn('X0',1,0.0024756917264312506).
nn('X0',2,0.1778842806816101).
nn('X0',3,0.5712596774101257).
nn('X0',4,0.0021903645247220993).
nn('X0',5,0.001295149209909141).
nn('X0',6,3.378479232196696e-05).
nn('X0',7,0.1307028830051422).
nn('X0',8,0.04958188161253929).
nn('X0',9,0.06431936472654343).
nn('X1',0,9.987016547086114e-09).
nn('X1',1,2.327002794014904e-16).
nn('X1',2,3.1944463740885e-10).
nn('X1',3,3.3435132531857848e-18).
nn('X1',4,2.759597705903616e-08).
nn('X1',5,2.0540944660751848e-06).
nn('X1',6,0.9999979734420776).
nn('X1',7,5.962080262540519e-15).
nn('X1',8,1.9264507523930607e-12).
nn('X1',9,5.597994161591572e-15).
nn('X2',0,5.758352778981179e-13).
nn('X2',1,8.01808113237712e-08).
nn('X2',2,4.3413988350948785e-06).
nn('X2',3,3.4404379345431835e-09).
nn('X2',4,8.404695273078744e-10).
nn('X2',5,1.2177972052995756e-07).
nn('X2',6,1.9987306032476226e-08).
nn('X2',7,5.8195651035930496e-06).
nn('X2',8,0.9999889731407166).
nn('X2',9,6.017330065333226e-07).
nn('X3',0,5.870655350737053e-11).
nn('X3',1,1.796834924773505e-12).
nn('X3',2,7.146509233280085e-06).
nn('X3',3,4.86401949118305e-19).
nn('X3',4,0.9999607801437378).
nn('X3',5,9.320886420027819e-06).
nn('X3',6,2.213804327766411e-05).
nn('X3',7,4.141139514635128e-11).
nn('X3',8,4.895726836082193e-13).
nn('X3',9,6.301033863564953e-07).
nn('X4',0,1.9601277472247602e-06).
nn('X4',1,0.9999898672103882).
nn('X4',2,3.131056928395992e-06).
nn('X4',3,9.309898772041914e-13).
nn('X4',4,3.42351029303245e-07).
nn('X4',5,6.4658496512493e-07).
nn('X4',6,2.385888365097344e-06).
nn('X4',7,1.0413873496872839e-06).
nn('X4',8,4.065697396526957e-07).
nn('X4',9,1.9660727446080273e-07).
nn('X5',0,4.572803247171464e-11).
nn('X5',1,3.104263441855437e-07).
nn('X5',2,1.3870077964384109e-05).
nn('X5',3,0.9999833703041077).
nn('X5',4,1.041451149088159e-15).
nn('X5',5,2.5070439733099192e-06).
nn('X5',6,2.5016850090637293e-17).
nn('X5',7,8.604856410743267e-11).
nn('X5',8,5.893072488327711e-14).
nn('X5',9,1.1293828285768859e-12).

a :- Pos=[f(['X0','X1','X2','X3'],21),f(['X4','X5'],4)], metaabd(Pos).