:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,3.3728287524015244e-11).
nn('X0',1,5.1115285615299655e-11).
nn('X0',2,3.0992112442618236e-05).
nn('X0',3,2.0252295139005836e-18).
nn('X0',4,0.9999106526374817).
nn('X0',5,4.218100002617575e-05).
nn('X0',6,7.421738700941205e-06).
nn('X0',7,3.0612321566536593e-09).
nn('X0',8,1.1352389167607058e-10).
nn('X0',9,8.879915185389109e-06).
nn('X1',0,0.0008824421092867851).
nn('X1',1,4.262875154381618e-05).
nn('X1',2,0.018753044307231903).
nn('X1',3,0.9802016615867615).
nn('X1',4,1.2972764196206299e-08).
nn('X1',5,7.851814734749496e-05).
nn('X1',6,3.2887502499079346e-08).
nn('X1',7,3.545109575497918e-05).
nn('X1',8,4.807209734281059e-06).
nn('X1',9,1.3826027043251088e-06).
nn('X2',0,0.4406932294368744).
nn('X2',1,0.002445939229801297).
nn('X2',2,0.507832944393158).
nn('X2',3,0.0063600181601941586).
nn('X2',4,0.0004255014064256102).
nn('X2',5,0.00028267083689570427).
nn('X2',6,0.004534156061708927).
nn('X2',7,0.0027720495127141476).
nn('X2',8,0.025909598916769028).
nn('X2',9,0.008743854239583015).
nn('X3',0,9.765998052521974e-17).
nn('X3',1,5.494136804950762e-16).
nn('X3',2,1.8078356589512623e-08).
nn('X3',3,1.0648180084141302e-25).
nn('X3',4,1.0).
nn('X3',5,5.344393372297418e-08).
nn('X3',6,4.13536399568315e-10).
nn('X3',7,4.081498390672879e-14).
nn('X3',8,6.244973871644689e-20).
nn('X3',9,7.410499991422625e-10).
nn('X4',0,1.7275990360499094e-10).
nn('X4',1,3.36699130798479e-08).
nn('X4',2,6.969564248643367e-12).
nn('X4',3,2.0915787501962768e-07).
nn('X4',4,7.104119538730441e-14).
nn('X4',5,0.9999997615814209).
nn('X4',6,3.9011819490987476e-13).
nn('X4',7,6.846631706025619e-09).
nn('X4',8,1.0747815567494368e-14).
nn('X4',9,5.064649871364124e-13).

a :- Pos=[f(['X0','X1'],7),f(['X2','X3','X4'],11)], metaabd(Pos).
