:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.9999982118606567).
nn('X0',1,2.8183820294930273e-12).
nn('X0',2,1.4678397519674036e-06).
nn('X0',3,1.6194032226302113e-11).
nn('X0',4,3.9313723039806e-14).
nn('X0',5,2.7629776244708637e-09).
nn('X0',6,5.990572926606319e-09).
nn('X0',7,3.516239814871369e-07).
nn('X0',8,1.3803273191115295e-08).
nn('X0',9,7.269094215445193e-10).
nn('X1',0,5.487610792442865e-07).
nn('X1',1,0.00012128870002925396).
nn('X1',2,0.999526858329773).
nn('X1',3,7.242520950967446e-05).
nn('X1',4,8.234788850813857e-08).
nn('X1',5,6.78711501223006e-07).
nn('X1',6,8.472840704598639e-07).
nn('X1',7,2.657109871506691e-05).
nn('X1',8,0.0002502899442333728).
nn('X1',9,4.1416538465455233e-07).
nn('X2',0,8.10954418284382e-07).
nn('X2',1,1.3837760093338147e-07).
nn('X2',2,9.558381862007082e-06).
nn('X2',3,1.9324780220052878e-13).
nn('X2',4,5.1013030315516517e-05).
nn('X2',5,0.00011079847172368318).
nn('X2',6,0.9998276829719543).
nn('X2',7,3.662840170060022e-11).
nn('X2',8,4.0171865833826814e-10).
nn('X2',9,1.83551521204528e-10).
nn('X3',0,5.171201800102665e-10).
nn('X3',1,4.277786658235527e-08).
nn('X3',2,4.3089144696750736e-07).
nn('X3',3,2.9137948998059215e-11).
nn('X3',4,0.9971370697021484).
nn('X3',5,0.00030234066070988774).
nn('X3',6,2.4082098093458626e-07).
nn('X3',7,1.696327490208205e-05).
nn('X3',8,3.874011667015509e-10).
nn('X3',9,0.0025428186636418104).

a :- Pos=[f(['X0','X1','X2','X3'],12)], metaabd(Pos).
