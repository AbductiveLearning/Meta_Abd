:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,3.1431620256164956e-15).
nn('X0',1,3.0094479425840648e-12).
nn('X0',2,4.074863779734983e-10).
nn('X0',3,1.42558456359132e-11).
nn('X0',4,8.228094854761947e-13).
nn('X0',5,2.0659949306889303e-09).
nn('X0',6,1.1803486139883024e-11).
nn('X0',7,9.315016313848901e-07).
nn('X0',8,0.999998927116394).
nn('X0',9,1.3874972637495375e-07).
nn('X1',0,2.2789263531830528e-10).
nn('X1',1,1.0).
nn('X1',2,6.452604834998599e-13).
nn('X1',3,2.8340782541760167e-24).
nn('X1',4,4.031037842796214e-15).
nn('X1',5,1.2327261607325557e-12).
nn('X1',6,5.876441369102869e-13).
nn('X1',7,3.337075841353121e-12).
nn('X1',8,9.111114132305618e-16).
nn('X1',9,4.166650782266943e-15).
nn('X2',0,2.7750443609697584e-10).
nn('X2',1,1.0519158877286827e-08).
nn('X2',2,1.3763576056646798e-08).
nn('X2',3,6.905468552531602e-08).
nn('X2',4,4.696119049185654e-10).
nn('X2',5,6.713376876632537e-08).
nn('X2',6,5.8783857840552685e-15).
nn('X2',7,0.9999585747718811).
nn('X2',8,1.1897167279117937e-12).
nn('X2',9,4.12782137573231e-05).
nn('X3',0,0.0002538844710215926).
nn('X3',1,1.33777566588833e-05).
nn('X3',2,0.9997299313545227).
nn('X3',3,6.950157427354497e-08).
nn('X3',4,2.8936490981124052e-09).
nn('X3',5,5.662229796143947e-08).
nn('X3',6,1.1676942222038633e-06).
nn('X3',7,4.9654904188400906e-08).
nn('X3',8,1.4103077319305157e-06).
nn('X3',9,4.252421703654363e-08).
nn('X4',0,5.187895908420614e-07).
nn('X4',1,0.0004317221464589238).
nn('X4',2,0.9995486736297607).
nn('X4',3,1.7887770431546102e-10).
nn('X4',4,2.296666329337782e-12).
nn('X4',5,1.414903350732688e-11).
nn('X4',6,8.004767515501499e-09).
nn('X4',7,1.8054521206067875e-05).
nn('X4',8,9.9947919807164e-07).
nn('X4',9,2.1804642813538777e-10).
nn('X5',0,5.2070363381062634e-06).
nn('X5',1,2.382899867009458e-12).
nn('X5',2,1.4172662154976479e-08).
nn('X5',3,2.6430860131836445e-14).
nn('X5',4,1.3804729803723603e-07).
nn('X5',5,5.562078058574116e-06).
nn('X5',6,0.9999890327453613).
nn('X5',7,1.838060302919442e-12).
nn('X5',8,1.6455022067152214e-10).
nn('X5',9,1.576103505519788e-12).
nn('X6',0,6.295825406149591e-13).
nn('X6',1,1.324092302468216e-08).
nn('X6',2,1.2599574574778671e-06).
nn('X6',3,1.1505586883231445e-07).
nn('X6',4,3.0836653763799404e-07).
nn('X6',5,3.1670977023168234e-06).
nn('X6',6,2.8219580006094702e-09).
nn('X6',7,4.1623232391430065e-05).
nn('X6',8,0.9997973442077637).
nn('X6',9,0.00015606696251779795).

a :- Pos=[f(['X0','X1','X2'],16),f(['X3','X4','X5','X6'],18)], metaabd(Pos).
