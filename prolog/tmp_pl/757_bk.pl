:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,1.213065699746557e-20).
nn('X0',2,1.785958262789239e-13).
nn('X0',3,3.0920154680590307e-21).
nn('X0',4,1.6464754756746267e-25).
nn('X0',5,2.9387723401693225e-14).
nn('X0',6,5.158513319194341e-14).
nn('X0',7,7.172322949280713e-16).
nn('X0',8,8.252797515053808e-18).
nn('X0',9,9.876805993817084e-20).
nn('X1',0,3.015429911101819e-06).
nn('X1',1,0.0010981060331687331).
nn('X1',2,0.0024743073154240847).
nn('X1',3,0.9747537970542908).
nn('X1',4,2.9273538530105725e-05).
nn('X1',5,0.007452927064150572).
nn('X1',6,2.696781393751735e-07).
nn('X1',7,0.012810947373509407).
nn('X1',8,0.0006826756289228797).
nn('X1',9,0.000694767979439348).
nn('X2',0,1.255089032703438e-09).
nn('X2',1,1.9307683707045498e-13).
nn('X2',2,6.847140042454125e-13).
nn('X2',3,9.441674872490147e-13).
nn('X2',4,1.551422727265328e-12).
nn('X2',5,0.9999998807907104).
nn('X2',6,9.219500185508878e-08).
nn('X2',7,2.2259853682538022e-11).
nn('X2',8,2.923089287981684e-12).
nn('X2',9,1.61116484737045e-11).
nn('X3',0,6.420890485969721e-10).
nn('X3',1,4.412532064179686e-07).
nn('X3',2,0.00037361474824137986).
nn('X3',3,0.9996238946914673).
nn('X3',4,1.2421303643582373e-14).
nn('X3',5,2.012486902458477e-06).
nn('X3',6,1.738928316933567e-14).
nn('X3',7,5.259910906829646e-08).
nn('X3',8,6.524331630508584e-10).
nn('X3',9,2.696743939267776e-10).
nn('X4',0,1.4869685344143235e-12).
nn('X4',1,1.6992763533091315e-12).
nn('X4',2,1.4400829059013631e-05).
nn('X4',3,4.387073353839568e-19).
nn('X4',4,0.9999814629554749).
nn('X4',5,4.069497208547546e-06).
nn('X4',6,1.5086253357665669e-09).
nn('X4',7,1.1169844910119409e-10).
nn('X4',8,1.2518348716569067e-13).
nn('X4',9,9.156517677411102e-08).
nn('X5',0,0.006607812363654375).
nn('X5',1,1.0349015394650607e-10).
nn('X5',2,2.049447944685312e-09).
nn('X5',3,4.37996791613493e-11).
nn('X5',4,1.620521772327521e-11).
nn('X5',5,0.05712734907865524).
nn('X5',6,0.935874879360199).
nn('X5',7,2.159830508885463e-10).
nn('X5',8,0.00038994199712760746).
nn('X5',9,3.676935925872593e-12).
nn('X6',0,4.1429761097472317e-13).
nn('X6',1,1.037250667301759e-11).
nn('X6',2,1.194465940003095e-13).
nn('X6',3,9.861932234386805e-13).
nn('X6',4,4.4175288122329844e-15).
nn('X6',5,3.890765747777136e-12).
nn('X6',6,7.880898157742636e-22).
nn('X6',7,1.0).
nn('X6',8,2.917367089673267e-17).
nn('X6',9,5.7694379052009026e-08).

a :- Pos=[f(['X0','X1','X2','X3'],11),f(['X4','X5','X6'],17)], metaabd(Pos).
