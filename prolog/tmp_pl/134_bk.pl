:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.6245031986272807e-09).
nn('X0',1,3.686149739223765e-06).
nn('X0',2,7.356321293627843e-05).
nn('X0',3,0.9997530579566956).
nn('X0',4,1.889938650023737e-09).
nn('X0',5,0.00016188669542316347).
nn('X0',6,1.3569355057987953e-12).
nn('X0',7,6.310123353614472e-06).
nn('X0',8,1.7645632510721043e-08).
nn('X0',9,1.533403974462999e-06).
nn('X1',0,1.0).
nn('X1',1,2.8909590243627117e-15).
nn('X1',2,2.0434105962863214e-08).
nn('X1',3,7.790678176600033e-17).
nn('X1',4,4.136213500340307e-15).
nn('X1',5,4.483679486716463e-12).
nn('X1',6,5.359230281776206e-10).
nn('X1',7,1.4430265793050068e-12).
nn('X1',8,5.64667453420431e-13).
nn('X1',9,1.5663180117069658e-13).
nn('X2',0,1.8251964384319308e-09).
nn('X2',1,2.5626106889831135e-06).
nn('X2',2,0.9999973773956299).
nn('X2',3,7.699557950848618e-14).
nn('X2',4,2.7281632331317107e-21).
nn('X2',5,5.683779673256017e-17).
nn('X2',6,1.1609454750891122e-15).
nn('X2',7,1.0393335081460009e-08).
nn('X2',8,3.0595931604232574e-12).
nn('X2',9,5.665121125614781e-17).
nn('X3',0,2.1780565198636914e-11).
nn('X3',1,8.353270963823434e-09).
nn('X3',2,2.074436844523575e-09).
nn('X3',3,3.553062954964048e-09).
nn('X3',4,4.84963948110817e-07).
nn('X3',5,2.8305362320679706e-06).
nn('X3',6,3.7546937672709724e-13).
nn('X3',7,0.9994542598724365).
nn('X3',8,7.547921489214904e-12).
nn('X3',9,0.0005424723494797945).

a :- Pos=[f(['X0','X1','X2','X3'],12)], metaabd(Pos).
