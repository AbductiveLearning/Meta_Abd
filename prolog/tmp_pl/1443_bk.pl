:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,2.3333600676633814e-11).
nn('X0',1,2.0734171757991193e-12).
nn('X0',2,1.3195596525183984e-11).
nn('X0',3,2.5948905246764298e-08).
nn('X0',4,1.925853920781151e-10).
nn('X0',5,0.9999982714653015).
nn('X0',6,8.284585691076263e-12).
nn('X0',7,5.789912620457471e-07).
nn('X0',8,3.901018619245633e-09).
nn('X0',9,1.048907620315731e-06).
nn('X1',0,1.0755497896752786e-06).
nn('X1',1,9.022071753861383e-05).
nn('X1',2,0.0007469199481420219).
nn('X1',3,0.9943612813949585).
nn('X1',4,7.912623004813213e-06).
nn('X1',5,0.0010990246664732695).
nn('X1',6,4.402813225112823e-09).
nn('X1',7,0.002548025920987129).
nn('X1',8,1.086490556190256e-05).
nn('X1',9,0.001134521677158773).
nn('X2',0,2.256053344140696e-09).
nn('X2',1,6.405376923313e-11).
nn('X2',2,3.394003966761261e-10).
nn('X2',3,7.294028048221435e-09).
nn('X2',4,2.1334078167262183e-12).
nn('X2',5,6.852660799916421e-11).
nn('X2',6,1.177743064875451e-17).
nn('X2',7,0.9995881915092468).
nn('X2',8,2.5539197492929766e-12).
nn('X2',9,0.0004117663484066725).
nn('X3',0,3.5946987608781455e-09).
nn('X3',1,1.0425718954820695e-08).
nn('X3',2,1.0579012110767394e-09).
nn('X3',3,8.849445976011339e-08).
nn('X3',4,2.3157029627185466e-09).
nn('X3',5,0.9999969005584717).
nn('X3',6,4.4511766494892413e-10).
nn('X3',7,2.508227908037952e-06).
nn('X3',8,3.596305120368015e-08).
nn('X3',9,4.641029249796702e-07).
nn('X4',0,1.1433197270871442e-11).
nn('X4',1,2.9598414830189768e-09).
nn('X4',2,1.2439844987444104e-10).
nn('X4',3,9.848595550199235e-11).
nn('X4',4,3.2099184963697847e-13).
nn('X4',5,7.689285874690199e-10).
nn('X4',6,1.1174912290570885e-18).
nn('X4',7,0.9999998807907104).
nn('X4',8,4.308354735661378e-15).
nn('X4',9,1.703639043171279e-07).
nn('X5',0,0.9999985694885254).
nn('X5',1,9.250923811709999e-12).
nn('X5',2,1.4147964293442783e-06).
nn('X5',3,1.3573495626084675e-12).
nn('X5',4,5.481131905770917e-14).
nn('X5',5,1.5681768383846162e-10).
nn('X5',6,1.821132400436909e-08).
nn('X5',7,1.7083090497749254e-10).
nn('X5',8,1.0051451893033914e-09).
nn('X5',9,2.3695805734247344e-11).

a :- Pos=[f(['X0','X1','X2','X3'],20),f(['X4','X5'],7)], metaabd(Pos).
