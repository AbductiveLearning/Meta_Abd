:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,8.115983579637387e-12).
nn('X0',1,1.8381164645919768e-12).
nn('X0',2,3.166703010926142e-11).
nn('X0',3,1.0157576468827756e-07).
nn('X0',4,5.448468982649501e-07).
nn('X0',5,1.5009005949195853e-07).
nn('X0',6,3.280652117153282e-14).
nn('X0',7,0.0004302227753214538).
nn('X0',8,9.379478305149291e-10).
nn('X0',9,0.9995690584182739).
nn('X1',0,5.830213822832775e-08).
nn('X1',1,0.9999998807907104).
nn('X1',2,1.0623314450342036e-09).
nn('X1',3,1.0391602856165832e-17).
nn('X1',4,2.882628510891827e-11).
nn('X1',5,4.3247516678945885e-09).
nn('X1',6,2.7632307553204782e-09).
nn('X1',7,4.5864148590091247e-10).
nn('X1',8,6.583063087706176e-12).
nn('X1',9,3.098643563959058e-11).
nn('X2',0,3.154241994707263e-06).
nn('X2',1,0.8049356937408447).
nn('X2',2,4.824609641218558e-05).
nn('X2',3,6.568311050614284e-07).
nn('X2',4,2.4760726091699325e-07).
nn('X2',5,5.230064289207803e-06).
nn('X2',6,1.3697906697807127e-10).
nn('X2',7,0.19497287273406982).
nn('X2',8,8.776627424822436e-08).
nn('X2',9,3.364983422216028e-05).
nn('X3',0,1.3556724631769157e-09).
nn('X3',1,1.7284448442552846e-15).
nn('X3',2,1.0).
nn('X3',3,9.124523749470095e-27).
nn('X3',4,4.713153533379028e-31).
nn('X3',5,2.5146499354207563e-24).
nn('X3',6,1.460822416765785e-19).
nn('X3',7,1.8933377430404598e-20).
nn('X3',8,1.4003038374141972e-22).
nn('X3',9,1.8182433771061464e-28).
nn('X4',0,2.0790475189959995e-13).
nn('X4',1,8.270198761928249e-13).
nn('X4',2,3.671608510558144e-07).
nn('X4',3,3.100468851032281e-23).
nn('X4',4,0.9999982118606567).
nn('X4',5,1.3558375258071464e-06).
nn('X4',6,4.8342375436050133e-08).
nn('X4',7,2.7582554296134987e-12).
nn('X4',8,1.9724396992034803e-16).
nn('X4',9,2.1746933143163005e-08).
nn('X5',0,3.757805089321664e-08).
nn('X5',1,1.0).
nn('X5',2,3.3661997633771534e-09).
nn('X5',3,8.90286014689545e-19).
nn('X5',4,2.5318481686187333e-11).
nn('X5',5,1.9828630057339147e-11).
nn('X5',6,3.4387727040896543e-12).
nn('X5',7,4.427848088184305e-10).
nn('X5',8,1.0011560807807185e-12).
nn('X5',9,2.170292374634175e-12).

a :- Pos=[f(['X0','X1','X2','X3'],13),f(['X4','X5'],5)], metaabd(Pos).
