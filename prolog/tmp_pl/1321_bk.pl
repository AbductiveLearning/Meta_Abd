:- ['../arithmetic_bk.pl'].

user:clp_range('0..9').

nn('X0',0,1.273013276659185e-05).
nn('X0',1,0.0001567774743307382).
nn('X0',2,0.00014319337788037956).
nn('X0',3,0.00014372915029525757).
nn('X0',4,1.1291665487078717e-06).
nn('X0',5,0.001502346363849938).
nn('X0',6,0.0010065712267532945).
nn('X0',7,0.0005122221191413701).
nn('X0',8,0.9964792132377625).
nn('X0',9,4.206028825137764e-05).
nn('X1',0,1.001799421196381e-09).
nn('X1',1,5.841038728249259e-06).
nn('X1',2,9.90163825917989e-05).
nn('X1',3,0.9998481869697571).
nn('X1',4,2.2930784007724192e-11).
nn('X1',5,4.700011049862951e-05).
nn('X1',6,5.0733776988526325e-14).
nn('X1',7,1.5824536347963658e-08).
nn('X1',8,3.237123347155091e-10).
nn('X1',9,2.2714274905411003e-10).
nn('X2',0,3.7595455637529085e-09).
nn('X2',1,2.9608706881845137e-06).
nn('X2',2,0.00016414055426139385).
nn('X2',3,0.9997835755348206).
nn('X2',4,1.9279551913276016e-11).
nn('X2',5,4.932258161716163e-05).
nn('X2',6,2.1500286089253667e-13).
nn('X2',7,4.063001313170389e-08).
nn('X2',8,2.9807734058806545e-10).
nn('X2',9,4.405431852649855e-10).
nn('X3',0,3.19436080505834e-14).
nn('X3',1,2.907826236742128e-11).
nn('X3',2,1.0692777774323758e-09).
nn('X3',3,2.454523528871988e-10).
nn('X3',4,2.421613083669616e-11).
nn('X3',5,6.147309932202916e-07).
nn('X3',6,7.638242260910033e-10).
nn('X3',7,1.1968533044637297e-06).
nn('X3',8,0.9999979734420776).
nn('X3',9,1.6974374261735647e-07).
nn('X4',0,1.7994388964992503e-11).
nn('X4',1,2.560908596294098e-09).
nn('X4',2,3.5914435869699446e-09).
nn('X4',3,9.766709990799427e-05).
nn('X4',4,0.002645734930410981).
nn('X4',5,4.44515171693638e-05).
nn('X4',6,4.303295914054317e-11).
nn('X4',7,0.00306625384837389).
nn('X4',8,1.539401068839652e-06).
nn('X4',9,0.9941443204879761).
nn('X5',0,3.5889705800005864e-17).
nn('X5',1,4.1950177462639e-14).
nn('X5',2,8.208016488708836e-09).
nn('X5',3,2.3760492930447204e-25).
nn('X5',4,0.9999998807907104).
nn('X5',5,1.3931509101894335e-07).
nn('X5',6,1.0827320290340836e-10).
nn('X5',7,2.9162920857491934e-14).
nn('X5',8,6.50019535731728e-18).
nn('X5',9,4.361240257821919e-09).

a :- Pos=[f(['X0','X1'],11),f(['X2','X3','X4','X5'],24)], metaabd(Pos).