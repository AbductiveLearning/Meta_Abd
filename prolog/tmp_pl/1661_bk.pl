:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,1.3244688013698075e-19).
nn('X0',2,1.0617444423655087e-12).
nn('X0',3,7.786417834553867e-20).
nn('X0',4,6.190473171264347e-26).
nn('X0',5,1.2040407081916756e-17).
nn('X0',6,2.0833423847319758e-17).
nn('X0',7,5.354131173791396e-14).
nn('X0',8,1.3238632793120245e-18).
nn('X0',9,1.2163575929487127e-17).
nn('X1',0,4.967453184900883e-11).
nn('X1',1,7.458867779062267e-14).
nn('X1',2,9.22689984313163e-13).
nn('X1',3,2.7916674505933103e-13).
nn('X1',4,2.918913721838834e-10).
nn('X1',5,1.0).
nn('X1',6,1.4822731930763666e-09).
nn('X1',7,1.6956099024856286e-10).
nn('X1',8,1.3134177773155287e-11).
nn('X1',9,4.024827138238152e-09).
nn('X2',0,7.410503766180909e-09).
nn('X2',1,1.0080074229769174e-17).
nn('X2',2,1.919452661050536e-12).
nn('X2',3,2.3728305950065817e-20).
nn('X2',4,8.325367617523227e-10).
nn('X2',5,1.0488702173461206e-05).
nn('X2',6,0.9999895691871643).
nn('X2',7,1.5507982124144163e-16).
nn('X2',8,6.411540000089352e-14).
nn('X2',9,8.218275628497275e-17).
nn('X3',0,2.7465547377906463e-19).
nn('X3',1,5.830403744083212e-19).
nn('X3',2,8.79478196883019e-18).
nn('X3',3,1.1821003065293212e-09).
nn('X3',4,8.008891327904166e-09).
nn('X3',5,2.8704951748892427e-09).
nn('X3',6,6.553513593181193e-22).
nn('X3',7,0.00023674432304687798).
nn('X3',8,8.208238776175047e-12).
nn('X3',9,0.9997633099555969).
nn('X4',0,1.29413896265973e-10).
nn('X4',1,4.170433963446296e-12).
nn('X4',2,5.759487071294034e-12).
nn('X4',3,5.835232991246997e-11).
nn('X4',4,3.513189088602542e-13).
nn('X4',5,1.0379181542008453e-10).
nn('X4',6,1.6562643609003867e-18).
nn('X4',7,0.9999881982803345).
nn('X4',8,3.3063857641374428e-12).
nn('X4',9,1.1814692697953433e-05).

a :- Pos=[f(['X0','X1','X2','X3','X4'],27)], metaabd(Pos).
