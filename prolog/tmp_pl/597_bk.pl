:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,1.2156438187981407e-16).
nn('X0',2,4.90930240903964e-10).
nn('X0',3,7.122219752331202e-21).
nn('X0',4,1.1578298706319385e-26).
nn('X0',5,1.6878895684852387e-16).
nn('X0',6,4.618223951722471e-15).
nn('X0',7,5.136929564445586e-15).
nn('X0',8,2.79412139702399e-18).
nn('X0',9,5.696012873184458e-20).
nn('X1',0,1.2844063377974446e-13).
nn('X1',1,1.8520941447164319e-09).
nn('X1',2,5.399303972808411e-07).
nn('X1',3,2.350399697093053e-08).
nn('X1',4,2.3851407515707024e-09).
nn('X1',5,2.539033161497173e-08).
nn('X1',6,3.54447166062144e-10).
nn('X1',7,2.1785057469969615e-05).
nn('X1',8,0.9999545812606812).
nn('X1',9,2.2964139134273864e-05).
nn('X2',0,1.4275328315271896e-11).
nn('X2',1,1.2112194060220904e-11).
nn('X2',2,2.6778768642543582e-06).
nn('X2',3,6.949791181297002e-18).
nn('X2',4,0.9999846816062927).
nn('X2',5,7.160741006373428e-06).
nn('X2',6,9.395320148541941e-07).
nn('X2',7,5.208573128356875e-09).
nn('X2',8,7.950016784209063e-14).
nn('X2',9,4.614749286702136e-06).
nn('X3',0,2.196634625306615e-07).
nn('X3',1,2.0117777239647694e-05).
nn('X3',2,0.0009794142097234726).
nn('X3',3,3.494047065566441e-11).
nn('X3',4,0.9981028437614441).
nn('X3',5,0.00031456357100978494).
nn('X3',6,0.0001613891072338447).
nn('X3',7,4.371093018562533e-05).
nn('X3',8,3.625809483764897e-07).
nn('X3',9,0.00037740517291240394).
nn('X4',0,0.9999200701713562).
nn('X4',1,5.832418992213206e-07).
nn('X4',2,6.799515540478751e-05).
nn('X4',3,2.28683028069554e-09).
nn('X4',4,8.760165925991714e-09).
nn('X4',5,1.2500200909926207e-06).
nn('X4',6,9.9153812698205e-06).
nn('X4',7,1.8622330344442162e-08).
nn('X4',8,7.989914507788853e-08).
nn('X4',9,1.9765669101445837e-09).
nn('X5',0,1.0469273226121345e-09).
nn('X5',1,2.2349418031808455e-06).
nn('X5',2,5.8797500969376415e-05).
nn('X5',3,0.9999350905418396).
nn('X5',4,1.5126400134153205e-15).
nn('X5',5,3.991840458184015e-06).
nn('X5',6,7.33569411134503e-15).
nn('X5',7,4.531877095814707e-09).
nn('X5',8,4.509591658430345e-11).
nn('X5',9,1.933197742512438e-12).

a :- Pos=[f(['X0','X1','X2','X3'],16),f(['X4','X5'],3)], metaabd(Pos).