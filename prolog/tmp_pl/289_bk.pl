:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,4.5398198795187567e-20).
nn('X0',2,2.4597374918139403e-12).
nn('X0',3,5.6167574998332275e-21).
nn('X0',4,5.35365186493671e-25).
nn('X0',5,1.4045238280552751e-16).
nn('X0',6,1.5287318394681393e-15).
nn('X0',7,3.5092090501899834e-14).
nn('X0',8,4.626240013454935e-18).
nn('X0',9,2.86640217602816e-17).
nn('X1',0,1.0079859863454388e-10).
nn('X1',1,1.9071064727427256e-09).
nn('X1',2,7.57811058349489e-09).
nn('X1',3,1.0938722283526658e-07).
nn('X1',4,3.66856518496661e-08).
nn('X1',5,1.630321094125975e-06).
nn('X1',6,1.994783893621843e-09).
nn('X1',7,2.8110798666602932e-05).
nn('X1',8,0.9998010993003845).
nn('X1',9,0.0001690343488007784).
nn('X2',0,1.8514628052912485e-08).
nn('X2',1,2.1043696563083358e-07).
nn('X2',2,0.9999997615814209).
nn('X2',3,4.165477302821817e-14).
nn('X2',4,1.3040465564425862e-23).
nn('X2',5,2.1814425312221922e-17).
nn('X2',6,1.237716623900861e-15).
nn('X2',7,9.027036063091742e-11).
nn('X2',8,1.9634781065033975e-14).
nn('X2',9,6.341578615174138e-19).
nn('X3',0,1.3812727850393003e-08).
nn('X3',1,3.103286587702314e-07).
nn('X3',2,0.00016268499894067645).
nn('X3',3,2.74998623872591e-09).
nn('X3',4,0.9420698881149292).
nn('X3',5,0.0006447891937568784).
nn('X3',6,4.0585791794001125e-06).
nn('X3',7,0.002274275291711092).
nn('X3',8,1.0851154002011754e-05).
nn('X3',9,0.05483299866318703).
nn('X4',0,1.998348437837194e-07).
nn('X4',1,0.00019705564773175865).
nn('X4',2,0.00014968932373449206).
nn('X4',3,0.994601845741272).
nn('X4',4,6.379194505967689e-09).
nn('X4',5,0.005050249397754669).
nn('X4',6,2.594117365362081e-09).
nn('X4',7,8.142370688801748e-07).
nn('X4',8,8.547714713813548e-08).
nn('X4',9,2.427892731304837e-08).
nn('X5',0,2.556160394462381e-09).
nn('X5',1,8.501181270048619e-08).
nn('X5',2,3.5168804402019305e-07).
nn('X5',3,5.894802370676189e-07).
nn('X5',4,2.660228259898645e-09).
nn('X5',5,2.0375662643346004e-05).
nn('X5',6,4.7064553643227924e-11).
nn('X5',7,0.9999691247940063).
nn('X5',8,2.0291072641565044e-11).
nn('X5',9,9.492014214629307e-06).
nn('X6',0,2.8232984838894026e-09).
nn('X6',1,1.0).
nn('X6',2,1.5858713647842748e-11).
nn('X6',3,4.221128111355459e-21).
nn('X6',4,2.60951579355434e-13).
nn('X6',5,1.2758351666808387e-11).
nn('X6',6,7.276921578334983e-13).
nn('X6',7,4.8924180678922013e-11).
nn('X6',8,7.733066913055177e-15).
nn('X6',9,3.495127974474828e-14).

a :- Pos=[f(['X0','X1','X2','X3'],14),f(['X4','X5','X6'],11)], metaabd(Pos).