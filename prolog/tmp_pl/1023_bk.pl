:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,3.148451243446286e-17).
nn('X0',1,9.191887702375556e-14).
nn('X0',2,2.7356872323025527e-10).
nn('X0',3,8.262965832395164e-12).
nn('X0',4,3.22218814130526e-12).
nn('X0',5,3.151165717252269e-10).
nn('X0',6,6.294452941724496e-14).
nn('X0',7,7.840641842449259e-07).
nn('X0',8,0.9999960660934448).
nn('X0',9,3.1837503229326103e-06).
nn('X1',0,1.2159937501365903e-09).
nn('X1',1,1.0).
nn('X1',2,1.755671857894825e-10).
nn('X1',3,3.279226675908633e-21).
nn('X1',4,2.3183113415092826e-12).
nn('X1',5,1.0462610595604604e-12).
nn('X1',6,3.1632032063133586e-14).
nn('X1',7,2.1836107921835435e-10).
nn('X1',8,7.760402360328968e-15).
nn('X1',9,5.97146098302477e-14).
nn('X2',0,0.002608560025691986).
nn('X2',1,0.0005890518077649176).
nn('X2',2,0.9967347979545593).
nn('X2',3,6.398381083272398e-05).
nn('X2',4,2.8253493988827927e-10).
nn('X2',5,5.89818739626935e-07).
nn('X2',6,1.0076064427266829e-06).
nn('X2',7,1.6301003142871195e-06).
nn('X2',8,3.6420317428564886e-07).
nn('X2',9,1.5963834698595747e-08).
nn('X3',0,1.0).
nn('X3',1,1.666349228738337e-23).
nn('X3',2,1.915303785254667e-14).
nn('X3',3,1.37021909837559e-25).
nn('X3',4,1.203525184520221e-34).
nn('X3',5,7.49201068361432e-20).
nn('X3',6,2.877991152185908e-19).
nn('X3',7,5.614238599158684e-18).
nn('X3',8,6.114582045403104e-23).
nn('X3',9,1.1945981997936636e-24).
nn('X4',0,1.0).
nn('X4',1,1.5929135375848474e-15).
nn('X4',2,8.577172749824058e-09).
nn('X4',3,3.3707138468586723e-18).
nn('X4',4,9.05086864923436e-17).
nn('X4',5,5.571624975622691e-13).
nn('X4',6,3.455008501518364e-10).
nn('X4',7,9.42048522523109e-13).
nn('X4',8,1.9138603896129425e-11).
nn('X4',9,1.4101575267731759e-13).

a :- Pos=[f(['X0','X1','X2'],11),f(['X3','X4'],0)], metaabd(Pos).
