:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,6.295610249830006e-20).
nn('X0',2,4.588521229642595e-11).
nn('X0',3,1.8949273192214967e-20).
nn('X0',4,1.021394178937704e-21).
nn('X0',5,2.791027093685186e-14).
nn('X0',6,1.8051815738832216e-13).
nn('X0',7,2.0532302258142315e-14).
nn('X0',8,1.0799346087390005e-15).
nn('X0',9,1.6923124039588863e-16).
nn('X1',0,1.7294766507802706e-08).
nn('X1',1,4.955123117378335e-11).
nn('X1',2,1.7792038065067572e-08).
nn('X1',3,4.1259642322719406e-14).
nn('X1',4,1.489389278575004e-09).
nn('X1',5,1.2344658898655325e-05).
nn('X1',6,0.9999876618385315).
nn('X1',7,2.7426381478401485e-12).
nn('X1',8,1.6582222261973811e-09).
nn('X1',9,5.90448100568769e-14).
nn('X2',0,3.067726073169297e-09).
nn('X2',1,1.0).
nn('X2',2,8.524261296827262e-10).
nn('X2',3,4.25309896514806e-21).
nn('X2',4,1.7856961569140406e-13).
nn('X2',5,1.1093736606432314e-12).
nn('X2',6,8.648681272017955e-13).
nn('X2',7,3.0681926443953955e-10).
nn('X2',8,2.744583531798239e-13).
nn('X2',9,1.0271439325167456e-13).
nn('X3',0,4.3301660035410805e-09).
nn('X3',1,1.766983892537155e-09).
nn('X3',2,1.0).
nn('X3',3,1.2323806255001846e-12).
nn('X3',4,2.197853267247604e-19).
nn('X3',5,7.987581573101136e-16).
nn('X3',6,6.7429709480425724e-15).
nn('X3',7,1.3642544559322456e-11).
nn('X3',8,8.324481512098081e-13).
nn('X3',9,8.413070045308613e-17).
nn('X4',0,4.561336197639321e-09).
nn('X4',1,7.556142032694708e-17).
nn('X4',2,4.4945644428473486e-11).
nn('X4',3,1.8922989528251095e-19).
nn('X4',4,2.7497450982849614e-09).
nn('X4',5,5.904344675400353e-07).
nn('X4',6,0.9999993443489075).
nn('X4',7,4.794523695298087e-15).
nn('X4',8,8.64330037663591e-14).
nn('X4',9,1.129880859385916e-15).

a :- Pos=[f(['X0','X1'],6),f(['X2','X3','X4'],9)], metaabd(Pos).
