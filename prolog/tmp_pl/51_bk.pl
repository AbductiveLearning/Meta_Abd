:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,3.9629681480857215e-11).
nn('X0',1,1.9321156951917703e-14).
nn('X0',2,4.213831025648776e-14).
nn('X0',3,6.620833032213239e-14).
nn('X0',4,4.185604770765017e-13).
nn('X0',5,1.0).
nn('X0',6,2.1041848263791962e-09).
nn('X0',7,1.1028346985630932e-10).
nn('X0',8,1.696076501467303e-09).
nn('X0',9,1.1749336226163365e-10).
nn('X1',0,1.4879530984071576e-12).
nn('X1',1,1.8235962903850964e-16).
nn('X1',2,4.915125164511869e-12).
nn('X1',3,2.820924827062754e-09).
nn('X1',4,3.7442174516399973e-07).
nn('X1',5,5.498451138841176e-10).
nn('X1',6,5.087840511141292e-16).
nn('X1',7,0.00014378468040376902).
nn('X1',8,9.338167739070258e-11).
nn('X1',9,0.9998559355735779).
nn('X2',0,1.0).
nn('X2',1,1.4050874899434574e-22).
nn('X2',2,1.144169936746764e-12).
nn('X2',3,1.6830502155845455e-21).
nn('X2',4,1.225392248045581e-25).
nn('X2',5,2.7305232020375444e-16).
nn('X2',6,2.3625055023727103e-14).
nn('X2',7,2.1423947424876875e-16).
nn('X2',8,1.3949932780305658e-15).
nn('X2',9,2.5180655840160557e-18).
nn('X3',0,1.6933741076030628e-09).
nn('X3',1,6.716341371948431e-10).
nn('X3',2,1.8259362771777532e-11).
nn('X3',3,4.965864386363705e-10).
nn('X3',4,1.742563211781789e-11).
nn('X3',5,0.9999998807907104).
nn('X3',6,1.6791135237781418e-07).
nn('X3',7,1.256525439252698e-10).
nn('X3',8,6.546532760332013e-10).
nn('X3',9,5.1226155262096285e-11).
nn('X4',0,1.621266055434889e-12).
nn('X4',1,4.804392052650073e-15).
nn('X4',2,1.1027857446666012e-11).
nn('X4',3,2.4926134045699655e-08).
nn('X4',4,7.688987352594268e-06).
nn('X4',5,1.313850717110654e-08).
nn('X4',6,2.3115578258480797e-14).
nn('X4',7,0.000509641831740737).
nn('X4',8,1.785149632282934e-11).
nn('X4',9,0.9994825720787048).
nn('X5',0,2.135050635843072e-06).
nn('X5',1,0.006125756539404392).
nn('X5',2,0.979433536529541).
nn('X5',3,6.497576123365434e-06).
nn('X5',4,0.00017831838340498507).
nn('X5',5,2.5516867026453838e-05).
nn('X5',6,0.00010172249312745407).
nn('X5',7,0.009601890109479427).
nn('X5',8,0.004402807913720608).
nn('X5',9,0.00012179512123111635).
nn('X6',0,2.3999205041036475e-06).
nn('X6',1,7.720636196334452e-13).
nn('X6',2,3.951650562328268e-09).
nn('X6',3,6.569875397757522e-17).
nn('X6',4,6.985272082982164e-11).
nn('X6',5,5.7804008974926546e-05).
nn('X6',6,0.9999397993087769).
nn('X6',7,1.6142723415586355e-14).
nn('X6',8,2.0962103164379187e-08).
nn('X6',9,2.6097137620947723e-14).

a :- Pos=[f(['X0','X1','X2'],14),f(['X3','X4'],14),f(['X5','X6'],8)], metaabd(Pos).
