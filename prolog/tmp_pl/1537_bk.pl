:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,6.271502570598386e-07).
nn('X0',1,4.203847493045032e-05).
nn('X0',2,1.7670431873284542e-07).
nn('X0',3,4.480134884943254e-05).
nn('X0',4,7.829235926237743e-08).
nn('X0',5,0.9999047517776489).
nn('X0',6,6.186576229083585e-06).
nn('X0',7,9.188328249365441e-07).
nn('X0',8,2.2029431079317874e-07).
nn('X0',9,1.266281088874166e-07).
nn('X1',0,7.655237121290126e-13).
nn('X1',1,2.5893197308501925e-11).
nn('X1',2,3.760420419496313e-09).
nn('X1',3,5.899165184120037e-11).
nn('X1',4,1.428519981342205e-13).
nn('X1',5,3.604239684484867e-10).
nn('X1',6,2.5529042768646093e-10).
nn('X1',7,1.7294445342486142e-07).
nn('X1',8,0.9999996423721313).
nn('X1',9,1.6302051619732083e-07).
nn('X2',0,2.5514682033822567e-10).
nn('X2',1,5.828754681983905e-20).
nn('X2',2,1.0297077318414949e-13).
nn('X2',3,1.8397100901306118e-22).
nn('X2',4,1.0677603766440225e-11).
nn('X2',5,8.023752684493957e-07).
nn('X2',6,0.9999991059303284).
nn('X2',7,8.632965071692234e-19).
nn('X2',8,7.824098390929544e-16).
nn('X2',9,1.1843466028018267e-19).
nn('X3',0,3.5477599590744147e-13).
nn('X3',1,4.752012043411635e-13).
nn('X3',2,1.5386119028154566e-13).
nn('X3',3,8.789592317870465e-12).
nn('X3',4,2.4545547326848506e-15).
nn('X3',5,3.599993636527188e-10).
nn('X3',6,5.474269176972433e-21).
nn('X3',7,0.9999996423721313).
nn('X3',8,2.916454544021499e-17).
nn('X3',9,3.783302133797406e-07).
nn('X4',0,6.488710957430754e-11).
nn('X4',1,6.598843924621178e-07).
nn('X4',2,3.519228926052165e-07).
nn('X4',3,1.0799467098365767e-08).
nn('X4',4,1.1927313581239218e-09).
nn('X4',5,2.3796255845809355e-06).
nn('X4',6,6.069127067576119e-08).
nn('X4',7,2.2348172933561727e-05).
nn('X4',8,0.9999583959579468).
nn('X4',9,1.5848905604798347e-05).

a :- Pos=[f(['X0','X1'],13),f(['X2','X3','X4'],21)], metaabd(Pos).
