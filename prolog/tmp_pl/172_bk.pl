:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,4.245003903946554e-09).
nn('X0',1,1.0).
nn('X0',2,2.0897162472666153e-11).
nn('X0',3,3.614542681694037e-23).
nn('X0',4,2.062304489778167e-14).
nn('X0',5,1.2677276763073397e-12).
nn('X0',6,6.495006897762334e-13).
nn('X0',7,1.595798796924508e-12).
nn('X0',8,3.776089819518616e-16).
nn('X0',9,1.5076454836418355e-15).
nn('X1',0,2.639279239602388e-09).
nn('X1',1,1.0).
nn('X1',2,1.7974803936948724e-11).
nn('X1',3,6.444088866860687e-22).
nn('X1',4,5.2653749958199436e-14).
nn('X1',5,3.478093897960055e-12).
nn('X1',6,4.500693708778952e-13).
nn('X1',7,4.547243692698544e-10).
nn('X1',8,1.4445396444230825e-14).
nn('X1',9,1.055923537084287e-13).
nn('X2',0,2.313091044925386e-06).
nn('X2',1,4.6100443462648855e-09).
nn('X2',2,0.9999977350234985).
nn('X2',3,1.1967605725859976e-12).
nn('X2',4,7.11008010958027e-15).
nn('X2',5,1.8084657577889518e-13).
nn('X2',6,1.502021618193794e-09).
nn('X2',7,4.170929573943383e-11).
nn('X2',8,2.6535476593814167e-10).
nn('X2',9,1.4634958013255012e-13).
nn('X3',0,1.0).
nn('X3',1,5.900027277747859e-20).
nn('X3',2,1.5213988055484684e-11).
nn('X3',3,7.038253650014968e-20).
nn('X3',4,1.1934146520560379e-23).
nn('X3',5,2.169843115135215e-14).
nn('X3',6,7.758348813651644e-14).
nn('X3',7,3.710702264122373e-13).
nn('X3',8,2.1439708325745113e-14).
nn('X3',9,4.005911298630185e-16).
nn('X4',0,4.056567415826459e-15).
nn('X4',1,6.244723933852728e-16).
nn('X4',2,5.817710272944601e-14).
nn('X4',3,1.289147206762209e-07).
nn('X4',4,1.5127992014640768e-07).
nn('X4',5,1.8472372076416832e-08).
nn('X4',6,1.808823349225073e-18).
nn('X4',7,0.005762719549238682).
nn('X4',8,2.9554716313162643e-11).
nn('X4',9,0.9942370653152466).
nn('X5',0,1.0251834103858215e-12).
nn('X5',1,5.652019108813577e-13).
nn('X5',2,8.484824434162874e-07).
nn('X5',3,9.146546392130836e-20).
nn('X5',4,0.9999979734420776).
nn('X5',5,1.0275708746121381e-06).
nn('X5',6,1.1176105374488543e-07).
nn('X5',7,2.8108730620868272e-11).
nn('X5',8,6.190909177856583e-15).
nn('X5',9,9.345379936576137e-08).

a :- Pos=[f(['X0','X1','X2','X3'],4),f(['X4','X5'],13)], metaabd(Pos).
