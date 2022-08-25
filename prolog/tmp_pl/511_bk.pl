:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,3.861585937082676e-15).
nn('X0',2,2.5520405344536812e-08).
nn('X0',3,1.3957909183690833e-16).
nn('X0',4,1.68672450113607e-16).
nn('X0',5,3.3311550132891776e-12).
nn('X0',6,6.552978715212987e-10).
nn('X0',7,6.272623045977532e-12).
nn('X0',8,4.228780867143733e-11).
nn('X0',9,9.0170628125677e-13).
nn('X1',0,3.1354584280052222e-06).
nn('X1',1,1.9573030262537428e-12).
nn('X1',2,2.782874597428986e-09).
nn('X1',3,2.517041623753221e-15).
nn('X1',4,4.6237444428776087e-10).
nn('X1',5,1.2150363545515575e-05).
nn('X1',6,0.9999847412109375).
nn('X1',7,1.0029000538564793e-14).
nn('X1',8,3.5727161229459625e-09).
nn('X1',9,3.0920130940635974e-15).
nn('X2',0,3.837357168134936e-11).
nn('X2',1,7.125540608943215e-12).
nn('X2',2,3.9052955314033966e-11).
nn('X2',3,1.309927881720796e-07).
nn('X2',4,2.9060331144137308e-05).
nn('X2',5,1.9080646325164707e-06).
nn('X2',6,3.006354117474769e-13).
nn('X2',7,0.00025915028527379036).
nn('X2',8,2.103291052435452e-08).
nn('X2',9,0.9997097253799438).
nn('X3',0,2.4702535481556376e-13).
nn('X3',1,5.184111603537267e-13).
nn('X3',2,9.288835197166726e-15).
nn('X3',3,2.2091100379106193e-14).
nn('X3',4,3.0204604008904345e-12).
nn('X3',5,1.0).
nn('X3',6,2.504394608360272e-11).
nn('X3',7,3.620190898358938e-13).
nn('X3',8,6.509713151555199e-14).
nn('X3',9,6.621600490142043e-11).
nn('X4',0,0.9999909400939941).
nn('X4',1,1.2111004473597253e-13).
nn('X4',2,9.98858013190329e-07).
nn('X4',3,5.737146884836101e-18).
nn('X4',4,9.238985444748327e-12).
nn('X4',5,2.6752014492537057e-10).
nn('X4',6,8.084498404059559e-06).
nn('X4',7,5.618277381953113e-14).
nn('X4',8,8.980898164902273e-13).
nn('X4',9,9.014067026439851e-14).

a :- Pos=[f(['X0','X1','X2','X3','X4'],20)], metaabd(Pos).
