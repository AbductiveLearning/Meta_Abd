:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.8571368798347976e-07).
nn('X0',1,0.9999982118606567).
nn('X0',2,1.1582193337744684e-06).
nn('X0',3,2.1381093645795116e-15).
nn('X0',4,1.3550804922601856e-09).
nn('X0',5,2.863949077891448e-09).
nn('X0',6,1.0589192633858602e-08).
nn('X0',7,3.034349447261775e-07).
nn('X0',8,1.1954028877880774e-07).
nn('X0',9,7.772750443280074e-09).
nn('X1',0,4.4096264183515754e-14).
nn('X1',1,5.2723416909333665e-17).
nn('X1',2,3.4463859717448475e-14).
nn('X1',3,3.6330694008768205e-10).
nn('X1',4,2.7529699764272664e-07).
nn('X1',5,7.195807505411267e-09).
nn('X1',6,1.6700396726108344e-16).
nn('X1',7,0.00026415573665872216).
nn('X1',8,1.1986037717903741e-09).
nn('X1',9,0.9997356534004211).
nn('X2',0,4.376080831036688e-09).
nn('X2',1,4.54630344393081e-06).
nn('X2',2,0.9999954700469971).
nn('X2',3,6.455429520471102e-15).
nn('X2',4,1.5909375652288582e-18).
nn('X2',5,3.011974250960019e-16).
nn('X2',6,3.750456434130356e-13).
nn('X2',7,2.3948196314904635e-08).
nn('X2',8,1.461603234276243e-11).
nn('X2',9,1.456013108034292e-15).
nn('X3',0,0.0006034349789842963).
nn('X3',1,5.415057603386231e-05).
nn('X3',2,0.008139853365719318).
nn('X3',3,9.872275404632092e-05).
nn('X3',4,2.282537252540351e-06).
nn('X3',5,6.444317114073783e-05).
nn('X3',6,0.003845426719635725).
nn('X3',7,0.00031536645838059485).
nn('X3',8,0.9863535761833191).
nn('X3',9,0.0005227167275734246).
nn('X4',0,2.3705401530627057e-10).
nn('X4',1,1.0085928536796018e-08).
nn('X4',2,5.937644473319459e-12).
nn('X4',3,6.737582936011677e-08).
nn('X4',4,3.6376056247039434e-13).
nn('X4',5,0.9999998807907104).
nn('X4',6,1.554896073008294e-10).
nn('X4',7,1.104569352650131e-10).
nn('X4',8,6.821079752460449e-14).
nn('X4',9,3.189047875599982e-13).

a :- Pos=[f(['X0','X1','X2'],12),f(['X3','X4'],13)], metaabd(Pos).