:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,2.6221573537162958e-14).
nn('X0',1,8.681579760638769e-11).
nn('X0',2,6.117249817805259e-09).
nn('X0',3,1.2810205118896079e-09).
nn('X0',4,2.1829247298654764e-09).
nn('X0',5,2.2515740738526802e-07).
nn('X0',6,3.379970608952121e-11).
nn('X0',7,5.8557016018312424e-06).
nn('X0',8,0.9999737739562988).
nn('X0',9,2.0111774574615993e-05).
nn('X1',0,8.249662143190208e-15).
nn('X1',1,1.639702089384354e-11).
nn('X1',2,3.9527606077172095e-08).
nn('X1',3,5.088533887017958e-18).
nn('X1',4,0.9999758005142212).
nn('X1',5,1.0557656423770823e-05).
nn('X1',6,1.8656619305001954e-10).
nn('X1',7,5.269578107203188e-09).
nn('X1',8,2.0607917228156886e-13).
nn('X1',9,1.3644862519868184e-05).
nn('X2',0,1.0).
nn('X2',1,1.46238246121963e-13).
nn('X2',2,2.460721848507319e-09).
nn('X2',3,3.4968332212042402e-15).
nn('X2',4,1.0836144606602855e-17).
nn('X2',5,1.216946075160985e-11).
nn('X2',6,1.5854541637883024e-11).
nn('X2',7,3.007834425972078e-09).
nn('X2',8,2.194578199721242e-11).
nn('X2',9,1.1134783199640008e-11).
nn('X3',0,1.6231567201430153e-08).
nn('X3',1,1.0).
nn('X3',2,1.465987997661955e-09).
nn('X3',3,3.581512916053204e-19).
nn('X3',4,2.362534863922794e-12).
nn('X3',5,3.988721158920683e-11).
nn('X3',6,1.6370802283227626e-11).
nn('X3',7,4.8543866704164884e-09).
nn('X3',8,2.1168214750511005e-12).
nn('X3',9,4.179432407797057e-12).
nn('X4',0,1.063148701019312e-12).
nn('X4',1,9.757816776811978e-11).
nn('X4',2,3.513318702630386e-08).
nn('X4',3,1.43357803406019e-09).
nn('X4',4,4.312677385348573e-12).
nn('X4',5,4.820876853273148e-08).
nn('X4',6,4.3457957232817535e-09).
nn('X4',7,1.6851316786414827e-06).
nn('X4',8,0.9999979734420776).
nn('X4',9,2.6579866130305163e-07).

a :- Pos=[f(['X0','X1'],12),f(['X2','X3','X4'],9)], metaabd(Pos).
