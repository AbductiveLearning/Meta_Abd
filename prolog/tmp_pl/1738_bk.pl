:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0195700017333673e-18).
nn('X0',1,8.895383177328052e-21).
nn('X0',2,2.475871183687787e-17).
nn('X0',3,8.045470539785882e-12).
nn('X0',4,7.85379441481382e-08).
nn('X0',5,7.95487120619498e-10).
nn('X0',6,2.6925966882242396e-21).
nn('X0',7,1.5394796719192527e-05).
nn('X0',8,4.868115865931319e-14).
nn('X0',9,0.9999845027923584).
nn('X1',0,4.161407329839051e-12).
nn('X1',1,8.646344440421672e-08).
nn('X1',2,4.9822779146779794e-06).
nn('X1',3,0.9999933242797852).
nn('X1',4,3.5242824091209795e-17).
nn('X1',5,1.5682189768995158e-06).
nn('X1',6,2.551423916963941e-18).
nn('X1',7,4.665440256346187e-11).
nn('X1',8,1.46403105838553e-14).
nn('X1',9,1.120103968886739e-13).
nn('X2',0,2.7851590391619885e-12).
nn('X2',1,1.4039783224895075e-11).
nn('X2',2,2.2623293863266447e-11).
nn('X2',3,1.2324230524995983e-08).
nn('X2',4,3.049809516042501e-09).
nn('X2',5,0.9999954700469971).
nn('X2',6,2.250366282208205e-10).
nn('X2',7,2.4428104552498553e-06).
nn('X2',8,1.991592597505587e-07).
nn('X2',9,1.792035163816763e-06).
nn('X3',0,3.148142866393755e-07).
nn('X3',1,3.037776252767799e-08).
nn('X3',2,1.8706614355323836e-06).
nn('X3',3,6.628736622105169e-12).
nn('X3',4,9.638553954971485e-09).
nn('X3',5,4.499360147747211e-05).
nn('X3',6,0.9999526739120483).
nn('X3',7,2.826497647046011e-10).
nn('X3',8,6.17311499695461e-08).
nn('X3',9,1.2278506336671491e-11).
nn('X4',0,8.916157639760058e-06).
nn('X4',1,6.084952474338934e-05).
nn('X4',2,0.001295472146011889).
nn('X4',3,0.998310387134552).
nn('X4',4,4.144194178934413e-07).
nn('X4',5,0.00026515775243751705).
nn('X4',6,1.4619726540487932e-09).
nn('X4',7,1.7875912817544304e-05).
nn('X4',8,1.9655178107313986e-07).
nn('X4',9,4.071816510986537e-05).

a :- Pos=[f(['X0','X1','X2'],17),f(['X3','X4'],9)], metaabd(Pos).