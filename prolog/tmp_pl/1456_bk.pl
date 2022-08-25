:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,1.509795909712501e-14).
nn('X0',2,3.283009419874361e-08).
nn('X0',3,4.766894751108072e-15).
nn('X0',4,1.2653667207534995e-17).
nn('X0',5,5.916346839524023e-12).
nn('X0',6,1.527397132383701e-11).
nn('X0',7,1.4736807329995827e-09).
nn('X0',8,6.170971719732243e-12).
nn('X0',9,1.0655442457194186e-12).
nn('X1',0,1.0583397269670058e-08).
nn('X1',1,2.8653466870309785e-06).
nn('X1',2,0.00047411536797881126).
nn('X1',3,5.540094077938207e-11).
nn('X1',4,0.9983226656913757).
nn('X1',5,0.00048073980724439025).
nn('X1',6,4.698529664892703e-06).
nn('X1',7,3.606775862863287e-05).
nn('X1',8,3.653161684269435e-08).
nn('X1',9,0.0006788430036976933).
nn('X2',0,0.00195726053789258).
nn('X2',1,8.504169818479568e-05).
nn('X2',2,0.9966227412223816).
nn('X2',3,0.00043529662070795894).
nn('X2',4,1.517262937511532e-08).
nn('X2',5,4.444044918727741e-07).
nn('X2',6,6.394352112693014e-06).
nn('X2',7,0.00042572119855321944).
nn('X2',8,0.0003834520757663995).
nn('X2',9,8.366945257876068e-05).
nn('X3',0,4.6432486084128574e-14).
nn('X3',1,4.027395177606547e-15).
nn('X3',2,2.668745216136137e-13).
nn('X3',3,7.847504690516871e-08).
nn('X3',4,7.458475010935217e-07).
nn('X3',5,8.203325307931664e-08).
nn('X3',6,3.7910283043346297e-16).
nn('X3',7,0.000585312838666141).
nn('X3',8,7.211993668931882e-09).
nn('X3',9,0.9994138479232788).
nn('X4',0,2.8155922038308745e-09).
nn('X4',1,4.0314821645166083e-16).
nn('X4',2,8.244833982207211e-11).
nn('X4',3,1.3257533870117082e-18).
nn('X4',4,1.0378052583970288e-10).
nn('X4',5,1.424841258312881e-07).
nn('X4',6,0.9999998807907104).
nn('X4',7,1.5851969615562423e-15).
nn('X4',8,6.264701078858778e-14).
nn('X4',9,4.800628711735202e-18).
nn('X5',0,1.4146996907271614e-09).
nn('X5',1,2.3553107674610408e-18).
nn('X5',2,1.1109370368078686e-11).
nn('X5',3,3.8491744642581364e-22).
nn('X5',4,3.165320436315788e-11).
nn('X5',5,4.715493062690257e-08).
nn('X5',6,1.0).
nn('X5',7,2.813115531839739e-18).
nn('X5',8,1.1773713497605044e-15).
nn('X5',9,1.0231349433783178e-18).

a :- Pos=[f(['X0','X1','X2','X3'],15),f(['X4','X5'],12)], metaabd(Pos).
