:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.7243925731769139e-10).
nn('X0',1,8.175037891078318e-08).
nn('X0',2,5.978041031085013e-07).
nn('X0',3,9.60182078415528e-05).
nn('X0',4,0.01768505945801735).
nn('X0',5,0.0015089802909642458).
nn('X0',6,6.066786184533157e-09).
nn('X0',7,0.0019250096520408988).
nn('X0',8,0.00010230232146568596).
nn('X0',9,0.9786818027496338).
nn('X1',0,1.0).
nn('X1',1,3.586659919682457e-23).
nn('X1',2,1.2861880207800172e-14).
nn('X1',3,1.6803284750675775e-23).
nn('X1',4,2.066643802802136e-29).
nn('X1',5,1.1482426278582578e-16).
nn('X1',6,5.460913525781488e-16).
nn('X1',7,7.814191281820221e-17).
nn('X1',8,2.0379852334822165e-19).
nn('X1',9,3.459961600783635e-21).
nn('X2',0,6.205209501786157e-07).
nn('X2',1,0.0004327339120209217).
nn('X2',2,0.9995376467704773).
nn('X2',3,1.9979902177169606e-09).
nn('X2',4,5.270241166220602e-13).
nn('X2',5,4.374432413523088e-11).
nn('X2',6,8.168795639917903e-10).
nn('X2',7,2.87012335320469e-05).
nn('X2',8,2.1470189892625058e-07).
nn('X2',9,7.860991024788433e-11).
nn('X3',0,1.88947765877856e-07).
nn('X3',1,0.00037778003024868667).
nn('X3',2,6.44284809823148e-05).
nn('X3',3,0.9524387717247009).
nn('X3',4,4.2064478122938453e-08).
nn('X3',5,0.04711601138114929).
nn('X3',6,2.2856976755747382e-08).
nn('X3',7,1.631794134482334e-06).
nn('X3',8,1.1382669526938116e-06).
nn('X3',9,8.10614508850449e-08).
nn('X4',0,6.359811959555373e-05).
nn('X4',1,0.0010134060867130756).
nn('X4',2,0.00020901700190734118).
nn('X4',3,0.023742202669382095).
nn('X4',4,3.0319993129523937e-06).
nn('X4',5,0.939073920249939).
nn('X4',6,0.00022219867969397455).
nn('X4',7,0.0017043258994817734).
nn('X4',8,0.03386182337999344).
nn('X4',9,0.00010655963706085458).
nn('X5',0,7.054413303508866e-11).
nn('X5',1,3.888466835633153e-06).
nn('X5',2,4.85777036374202e-06).
nn('X5',3,0.9990236759185791).
nn('X5',4,9.25504117788023e-11).
nn('X5',5,0.0009675241308286786).
nn('X5',6,3.170283588600775e-13).
nn('X5',7,8.004217733059704e-09).
nn('X5',8,1.8069962193223432e-10).
nn('X5',9,1.730482201978134e-09).

a :- Pos=[f(['X0','X1'],9),f(['X2','X3','X4','X5'],13)], metaabd(Pos).
