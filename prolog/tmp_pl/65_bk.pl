:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,0.10661572962999344).
nn('X0',1,2.8392672902555205e-05).
nn('X0',2,0.0009438756969757378).
nn('X0',3,0.018848830834031105).
nn('X0',4,5.873976647308154e-07).
nn('X0',5,0.8656596541404724).
nn('X0',6,5.345874888007529e-05).
nn('X0',7,0.007489870768040419).
nn('X0',8,0.0002840641827788204).
nn('X0',9,7.567481225123629e-05).
nn('X1',0,2.204152015257161e-14).
nn('X1',1,1.0568881551452792e-14).
nn('X1',2,1.4400893671995263e-13).
nn('X1',3,7.988055727992105e-08).
nn('X1',4,1.0389927638243535e-06).
nn('X1',5,2.4790390185103206e-08).
nn('X1',6,2.3674547714700543e-17).
nn('X1',7,0.0010915208840742707).
nn('X1',8,8.315318572593711e-11).
nn('X1',9,0.9989073872566223).
nn('X2',0,5.782846073998371e-06).
nn('X2',1,2.200262997575919e-06).
nn('X2',2,0.014447896741330624).
nn('X2',3,2.857981733217496e-11).
nn('X2',4,0.9838393926620483).
nn('X2',5,0.001602635718882084).
nn('X2',6,2.654212585184723e-05).
nn('X2',7,2.29761099035386e-06).
nn('X2',8,2.7823918458125263e-07).
nn('X2',9,7.298809214262292e-05).
nn('X3',0,4.257469399249203e-08).
nn('X3',1,1.3143665000825422e-07).
nn('X3',2,1.653265144341276e-06).
nn('X3',3,0.00012378155952319503).
nn('X3',4,3.545368144841632e-06).
nn('X3',5,0.9826496839523315).
nn('X3',6,2.5718540541674884e-07).
nn('X3',7,0.0008216241258196533).
nn('X3',8,0.008776180446147919).
nn('X3',9,0.007623056881129742).
nn('X4',0,4.2911303206789615e-16).
nn('X4',1,9.126162723132968e-19).
nn('X4',2,4.769459566872358e-15).
nn('X4',3,1.0836607167163947e-08).
nn('X4',4,6.065012381206714e-10).
nn('X4',5,8.452416717930067e-12).
nn('X4',6,2.580697892156568e-21).
nn('X4',7,2.349083479202818e-05).
nn('X4',8,3.562655032096629e-11).
nn('X4',9,0.9999765753746033).
nn('X5',0,1.3062314630548494e-13).
nn('X5',1,3.876559837023841e-11).
nn('X5',2,2.3155439521360677e-07).
nn('X5',3,3.697532741442251e-17).
nn('X5',4,0.9999953508377075).
nn('X5',5,2.5648967039160198e-06).
nn('X5',6,5.493862698102703e-09).
nn('X5',7,4.765127403771885e-09).
nn('X5',8,6.566941407977131e-14).
nn('X5',9,1.8147211449104361e-06).

a :- Pos=[f(['X0','X1','X2','X3'],23),f(['X4','X5'],13)], metaabd(Pos).
