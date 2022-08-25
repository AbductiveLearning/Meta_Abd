:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.999955415725708).
nn('X0',1,1.14881307591852e-11).
nn('X0',2,4.451464337762445e-05).
nn('X0',3,8.282706118190042e-12).
nn('X0',4,2.475058223964325e-14).
nn('X0',5,4.209113960151889e-11).
nn('X0',6,1.8437628312995002e-08).
nn('X0',7,5.721226270516411e-10).
nn('X0',8,3.0915522586383304e-08).
nn('X0',9,4.2254429122312587e-11).
nn('X1',0,3.466011047506372e-14).
nn('X1',1,3.110076779444526e-11).
nn('X1',2,7.810528472873557e-07).
nn('X1',3,1.3144618061948625e-20).
nn('X1',4,0.9999978542327881).
nn('X1',5,1.1150793852721108e-06).
nn('X1',6,3.566542616795232e-08).
nn('X1',7,4.016037224996438e-10).
nn('X1',8,5.920807735152606e-15).
nn('X1',9,2.0254708488209872e-07).
nn('X2',0,1.7806644336815225e-07).
nn('X2',1,1.3694766494154464e-05).
nn('X2',2,0.9999861717224121).
nn('X2',3,4.1258704785995803e-10).
nn('X2',4,1.0473111083057215e-16).
nn('X2',5,5.318418231856081e-13).
nn('X2',6,1.5928946529852883e-12).
nn('X2',7,3.4122205505582315e-08).
nn('X2',8,1.7671415719622274e-10).
nn('X2',9,6.689569417069746e-14).
nn('X3',0,1.5110249051986102e-08).
nn('X3',1,5.9516205510590225e-06).
nn('X3',2,0.9999940991401672).
nn('X3',3,7.218099002618411e-14).
nn('X3',4,9.528739119553339e-19).
nn('X3',5,1.5228532190559576e-15).
nn('X3',6,2.6418323366590724e-13).
nn('X3',7,3.265566661525554e-08).
nn('X3',8,3.9114753103142164e-11).
nn('X3',9,2.140908808470187e-15).
nn('X4',0,6.853043942101067e-06).
nn('X4',1,2.775827124423813e-05).
nn('X4',2,0.9999637007713318).
nn('X4',3,7.528098977527264e-11).
nn('X4',4,6.339944628314109e-12).
nn('X4',5,1.0685101241303396e-11).
nn('X4',6,1.7272860919348432e-10).
nn('X4',7,1.6909012856558547e-06).
nn('X4',8,1.2731309340097141e-09).
nn('X4',9,3.324175310015498e-11).
nn('X5',0,2.358343338323965e-12).
nn('X5',1,3.204929724387462e-11).
nn('X5',2,5.405823358395345e-12).
nn('X5',3,4.107904294503584e-10).
nn('X5',4,7.932783661727949e-13).
nn('X5',5,1.0).
nn('X5',6,7.153866041220702e-12).
nn('X5',7,3.608135925453615e-12).
nn('X5',8,2.0900358402515043e-14).
nn('X5',9,9.4944350992332e-11).

a :- Pos=[f(['X0','X1'],4),f(['X2','X3','X4','X5'],11)], metaabd(Pos).