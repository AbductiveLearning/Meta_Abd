:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,2.085223229641997e-07).
nn('X0',1,1.2655977116082795e-05).
nn('X0',2,0.9999866485595703).
nn('X0',3,1.1709133662662907e-10).
nn('X0',4,6.216434356903414e-18).
nn('X0',5,1.385990660348474e-13).
nn('X0',6,4.892503638548665e-14).
nn('X0',7,4.339145220910723e-07).
nn('X0',8,1.440735660114545e-12).
nn('X0',9,2.7907928043719755e-14).
nn('X1',0,2.2937074390938506e-06).
nn('X1',1,0.0002752732834778726).
nn('X1',2,0.0013686111196875572).
nn('X1',3,0.9973139762878418).
nn('X1',4,6.213561846379889e-06).
nn('X1',5,0.000993749126791954).
nn('X1',6,1.0546802542421574e-08).
nn('X1',7,2.3752056222292595e-05).
nn('X1',8,3.705344170157332e-06).
nn('X1',9,1.2431076356733683e-05).
nn('X2',0,1.1753460704699847e-10).
nn('X2',1,8.266457451281545e-14).
nn('X2',2,5.756581599207153e-14).
nn('X2',3,4.6157327959761574e-11).
nn('X2',4,2.576386234345515e-16).
nn('X2',5,1.0).
nn('X2',6,5.7297961514324314e-11).
nn('X2',7,1.500482377236878e-11).
nn('X2',8,1.0959217203437149e-13).
nn('X2',9,7.499214465557513e-13).
nn('X3',0,1.294988464906055e-06).
nn('X3',1,0.009010516107082367).
nn('X3',2,0.0006286485004238784).
nn('X3',3,6.038738433744584e-07).
nn('X3',4,0.0012453267117962241).
nn('X3',5,0.9872378706932068).
nn('X3',6,4.393133713165298e-05).
nn('X3',7,0.0015842757420614362).
nn('X3',8,0.00019063548825215548).
nn('X3',9,5.6861674238462e-05).
nn('X4',0,3.276818446034009e-14).
nn('X4',1,5.348675307523189e-16).
nn('X4',2,1.4046303996231163e-13).
nn('X4',3,4.174618428720578e-09).
nn('X4',4,4.639049166144105e-06).
nn('X4',5,6.400648544513388e-08).
nn('X4',6,3.529141069515769e-16).
nn('X4',7,0.0002941162383649498).
nn('X4',8,8.368663401148169e-12).
nn('X4',9,0.9997011423110962).
nn('X5',0,0.9999998807907104).
nn('X5',1,7.270951825648006e-14).
nn('X5',2,1.530894309098585e-07).
nn('X5',3,1.5638217039674537e-14).
nn('X5',4,2.579641487840932e-16).
nn('X5',5,8.1872867860322e-12).
nn('X5',6,1.9289676800315902e-10).
nn('X5',7,8.097592041345081e-10).
nn('X5',8,1.2594311704638983e-10).
nn('X5',9,3.508369376264975e-12).
nn('X6',0,1.328162024805124e-08).
nn('X6',1,7.522969099227339e-05).
nn('X6',2,1.9048618923989125e-05).
nn('X6',3,0.9993165731430054).
nn('X6',4,1.592607712375127e-10).
nn('X6',5,0.0005890358006581664).
nn('X6',6,1.917710998680655e-12).
nn('X6',7,1.1655018283818208e-07).
nn('X6',8,3.8828962267700717e-10).
nn('X6',9,7.0036363375436395e-09).

a :- Pos=[f(['X0','X1','X2'],10),f(['X3','X4'],14),f(['X5','X6'],3)], metaabd(Pos).