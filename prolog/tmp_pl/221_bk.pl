:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.7560868327526867e-14).
nn('X0',1,2.818017417384618e-15).
nn('X0',2,1.7637279169698677e-13).
nn('X0',3,3.7143073061685072e-09).
nn('X0',4,3.1600711736246012e-06).
nn('X0',5,3.909156305326178e-07).
nn('X0',6,1.5723747885589435e-15).
nn('X0',7,0.00042290700366720557).
nn('X0',8,1.9552239827191897e-09).
nn('X0',9,0.9995735287666321).
nn('X1',0,7.180551619967446e-08).
nn('X1',1,1.7683223592010466e-13).
nn('X1',2,5.175563311254905e-10).
nn('X1',3,2.4444911342385222e-15).
nn('X1',4,1.129210058792296e-09).
nn('X1',5,6.339948595268652e-05).
nn('X1',6,0.9999364614486694).
nn('X1',7,7.361502358214922e-14).
nn('X1',8,3.2050656573190395e-10).
nn('X1',9,6.511621686191953e-15).
nn('X2',0,1.6951047143720643e-07).
nn('X2',1,5.6530202527937945e-06).
nn('X2',2,0.0018442142754793167).
nn('X2',3,0.9981469511985779).
nn('X2',4,4.881398237011059e-12).
nn('X2',5,2.749369059529272e-06).
nn('X2',6,7.997258454420231e-13).
nn('X2',7,1.5594100943872036e-07).
nn('X2',8,6.26596063924012e-09).
nn('X2',9,2.844767754694999e-10).
nn('X3',0,4.753496045135225e-14).
nn('X3',1,5.896724081244642e-11).
nn('X3',2,2.9749087308156286e-12).
nn('X3',3,2.1266097064744083e-13).
nn('X3',4,5.837447834681521e-15).
nn('X3',5,3.417282776196906e-11).
nn('X3',6,6.290737639034186e-20).
nn('X3',7,1.0).
nn('X3',8,9.365032011318122e-18).
nn('X3',9,2.2020868684080597e-08).

a :- Pos=[f(['X0','X1','X2','X3'],25)], metaabd(Pos).
