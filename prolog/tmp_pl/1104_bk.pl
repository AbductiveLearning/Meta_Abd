:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.0).
nn('X0',1,1.554286961758442e-23).
nn('X0',2,1.5319740918548486e-12).
nn('X0',3,5.4549782871991806e-24).
nn('X0',4,1.0010783554210636e-26).
nn('X0',5,4.568855599229987e-18).
nn('X0',6,2.563562949617473e-16).
nn('X0',7,7.902383293497154e-17).
nn('X0',8,6.436310440851008e-19).
nn('X0',9,8.81569870403406e-20).
nn('X1',0,1.0).
nn('X1',1,4.448058795463689e-19).
nn('X1',2,3.009222374322079e-13).
nn('X1',3,2.5269240841469105e-21).
nn('X1',4,1.978813946878088e-25).
nn('X1',5,4.208388966017609e-16).
nn('X1',6,3.4957512954328005e-16).
nn('X1',7,1.583225331435248e-13).
nn('X1',8,3.695131601560132e-19).
nn('X1',9,8.135115356486501e-18).
nn('X2',0,6.59277580444817e-12).
nn('X2',1,1.2655638670366898e-07).
nn('X2',2,4.6014331234189854e-10).
nn('X2',3,1.9983588395167118e-10).
nn('X2',4,1.7058476853293314e-10).
nn('X2',5,1.3334308102130876e-09).
nn('X2',6,1.691060515732624e-16).
nn('X2',7,0.9999949336051941).
nn('X2',8,1.4415933724531982e-12).
nn('X2',9,5.055527708464069e-06).
nn('X3',0,1.0).
nn('X3',1,1.3029396810940103e-20).
nn('X3',2,9.795647799848428e-12).
nn('X3',3,2.7545770730288965e-22).
nn('X3',4,1.5305802716069212e-24).
nn('X3',5,1.5562140606110303e-13).
nn('X3',6,1.5642813433469627e-10).
nn('X3',7,1.140711578514881e-16).
nn('X3',8,4.5548905359266145e-14).
nn('X3',9,1.546837438996965e-19).
nn('X4',0,1.3837191392696013e-09).
nn('X4',1,2.015867898972376e-15).
nn('X4',2,7.170782023724342e-11).
nn('X4',3,2.533145731783775e-16).
nn('X4',4,5.297457264519245e-11).
nn('X4',5,3.700036813825136e-06).
nn('X4',6,0.9999963045120239).
nn('X4',7,4.078340855133422e-13).
nn('X4',8,2.656392293631793e-11).
nn('X4',9,7.407876988560621e-16).
nn('X5',0,3.545590843145874e-08).
nn('X5',1,1.0).
nn('X5',2,2.252342756747794e-09).
nn('X5',3,2.850513401309027e-18).
nn('X5',4,1.1298142976734482e-11).
nn('X5',5,8.665400341723029e-11).
nn('X5',6,1.874009974978108e-11).
nn('X5',7,7.629743059567318e-09).
nn('X5',8,1.0765058549438988e-12).
nn('X5',9,1.720482090128428e-11).
nn('X6',0,1.9988104053847433e-12).
nn('X6',1,6.597481672088179e-08).
nn('X6',2,1.0236980415356811e-05).
nn('X6',3,0.9999884366989136).
nn('X6',4,2.9650644546103328e-15).
nn('X6',5,1.2694035831373185e-06).
nn('X6',6,2.5348252972021457e-18).
nn('X6',7,2.4542848309216936e-10).
nn('X6',8,4.765733299530797e-14).
nn('X6',9,3.0072059425956255e-11).

a :- Pos=[f(['X0','X1','X2'],7),f(['X3','X4','X5','X6'],10)], metaabd(Pos).